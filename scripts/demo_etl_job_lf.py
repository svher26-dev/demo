import sys
import boto3
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.window import Window

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read via Glue catalog — Lake Formation governs access to these tables
TABLE_BUCKET_ARN = "arn:aws:s3tables:us-east-1:256097482558:bucket/demo-bucket"
CATALOG   = "glue_catalog"
DATABASE  = "demo_database"

spark.conf.set(f"spark.sql.catalog.{CATALOG}", "org.apache.iceberg.spark.SparkCatalog")
spark.conf.set(f"spark.sql.catalog.{CATALOG}.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog")
spark.conf.set(f"spark.sql.catalog.{CATALOG}.warehouse", TABLE_BUCKET_ARN)
spark.conf.set(f"spark.sql.catalog.{CATALOG}.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")

# Aurora PostgreSQL connection (IAM auth)
AURORA_HOST = "database-1.cluster-cih4ce2i22u1.us-east-1.rds.amazonaws.com"
AURORA_PORT = 5432
AURORA_DB   = "postgres"
AURORA_USER = "postgres"
REGION      = "us-east-1"

rds_client = boto3.client("rds", region_name=REGION)
auth_token = rds_client.generate_db_auth_token(
    DBHostname=AURORA_HOST,
    Port=AURORA_PORT,
    DBUsername=AURORA_USER,
    Region=REGION
)

jdbc_url = (
    f"jdbc:postgresql://{AURORA_HOST}:{AURORA_PORT}/{AURORA_DB}"
    "?ssl=true&sslmode=require"
)
jdbc_props = {
    "user": AURORA_USER,
    "password": auth_token,
    "driver": "org.postgresql.Driver",
}

# ── Read source tables (via Glue catalog → Lake Formation enforces grants) ──
print("=== Reading source tables via Glue catalog (Lake Formation governed) ===")
customers = spark.table(f"{CATALOG}.{DATABASE}.demo_customers")
orders    = spark.table(f"{CATALOG}.{DATABASE}.demo_orders")
products  = spark.table(f"{CATALOG}.{DATABASE}.demo_products")

print(f"  customers : {customers.count():,}")
print(f"  orders    : {orders.count():,}")
print(f"  products  : {products.count():,}")

# ── curated_customer_summary ────────────────────────────────────────────────
print("=== Building curated_customer_summary ===")

orders_with_cat = orders.join(
    products.select("product_id", "category"),
    on="product_id", how="left"
)

cat_counts = orders_with_cat.groupBy("customer_id", "category").agg(
    F.count("*").alias("cnt")
)
w_cat = Window.partitionBy("customer_id").orderBy(F.desc("cnt"))
fav_cat = (
    cat_counts
    .withColumn("rn", F.row_number().over(w_cat))
    .filter(F.col("rn") == 1)
    .select("customer_id", F.col("category").alias("favourite_category"))
)

order_agg = orders_with_cat.groupBy("customer_id").agg(
    F.count("order_id").alias("total_orders"),
    F.round(F.sum("amount"), 2).alias("total_spend"),
    F.round(F.avg("amount"), 2).alias("avg_order_value"),
    F.max("order_date").alias("last_order_date"),
)

customer_summary = (
    customers.select("customer_id", "first_name", "last_name", "city")
    .join(order_agg, on="customer_id", how="left")
    .join(fav_cat,   on="customer_id", how="left")
    .fillna({"total_orders": 0, "total_spend": 0.0, "avg_order_value": 0.0})
)

print(f"  rows: {customer_summary.count():,}")
customer_summary.write.jdbc(
    url=jdbc_url, table="curated_customer_summary",
    mode="overwrite", properties=jdbc_props
)
print("  Written: curated_customer_summary")

# ── curated_category_revenue ────────────────────────────────────────────────
print("=== Building curated_category_revenue ===")

category_revenue = (
    orders_with_cat
    .filter(F.col("category").isNotNull())
    .groupBy("category").agg(
        F.count("order_id").alias("total_orders"),
        F.sum("quantity").alias("total_units_sold"),
        F.round(F.sum("amount"), 2).alias("total_revenue"),
        F.round(F.avg("amount"), 2).alias("avg_order_value"),
    )
)

print(f"  rows: {category_revenue.count():,}")
category_revenue.write.jdbc(
    url=jdbc_url, table="curated_category_revenue",
    mode="overwrite", properties=jdbc_props
)
print("  Written: curated_category_revenue")

# ── curated_daily_orders ────────────────────────────────────────────────────
print("=== Building curated_daily_orders ===")

daily_orders = (
    orders
    .withColumn("order_date", F.to_date("order_date"))
    .groupBy("order_date").agg(
        F.count("order_id").alias("total_orders"),
        F.round(F.sum("amount"), 2).alias("total_revenue"),
        F.sum(F.when(F.col("status") == "completed",  1).otherwise(0)).alias("completed_count"),
        F.sum(F.when(F.col("status") == "pending",    1).otherwise(0)).alias("pending_count"),
        F.sum(F.when(F.col("status") == "shipped",    1).otherwise(0)).alias("shipped_count"),
        F.sum(F.when(F.col("status") == "cancelled",  1).otherwise(0)).alias("cancelled_count"),
        F.sum(F.when(F.col("status") == "returned",   1).otherwise(0)).alias("returned_count"),
    )
    .orderBy("order_date")
)

print(f"  rows: {daily_orders.count():,}")
daily_orders.write.jdbc(
    url=jdbc_url, table="curated_daily_orders",
    mode="overwrite", properties=jdbc_props
)
print("  Written: curated_daily_orders")

job.commit()
print("=== ETL complete (Lake Formation governed) — 3 curated tables written to Aurora ===")
