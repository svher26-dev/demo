import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

TABLE_BUCKET_ARN = "arn:aws:s3tables:us-east-1:256097482558:bucket/demo-bucket"
CATALOG = "s3tablesbucket"
NAMESPACE = "demo"

# Configure S3 Tables Iceberg catalog
spark.conf.set(f"spark.sql.catalog.{CATALOG}", "org.apache.iceberg.spark.SparkCatalog")
spark.conf.set(f"spark.sql.catalog.{CATALOG}.catalog-impl", "software.amazon.s3tables.iceberg.S3TablesCatalog")
spark.conf.set(f"spark.sql.catalog.{CATALOG}.warehouse", TABLE_BUCKET_ARN)

print("=== Loading demo_customers (100,000 records) ===")
first_names = F.array([F.lit(n) for n in ["James","Mary","John","Patricia","Robert","Jennifer","Michael","Linda","William","Barbara","David","Susan","Richard","Jessica","Joseph","Sarah","Thomas","Karen","Charles","Lisa"]])
last_names  = F.array([F.lit(n) for n in ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson","Anderson","Thomas","Taylor","Moore","Jackson","Martin"]])
cities      = F.array([F.lit(c) for c in ["New York","Los Angeles","Chicago","Houston","Phoenix","Philadelphia","San Antonio","San Diego","Dallas","San Jose","Austin","Jacksonville","Fort Worth","Columbus","Charlotte","Indianapolis","San Francisco","Seattle","Denver","Nashville"]])

customers_df = spark.range(1, 100001).select(
    F.concat(F.lit("CUST-"), F.lpad(F.col("id").cast("string"), 6, "0")).alias("customer_id"),
    F.element_at(first_names, (F.col("id") % 20 + 1).cast("integer")).alias("first_name"),
    F.element_at(last_names,  (F.col("id") % 20 + 1).cast("integer")).alias("last_name"),
    F.concat(F.lit("user"), F.col("id").cast("string"), F.lit("@example.com")).alias("email"),
    F.concat(F.lit("+1-555-"), F.lpad((F.col("id") % 9000 + 1000).cast("string"), 4, "0")).alias("phone"),
    F.element_at(cities, (F.col("id") % 20 + 1).cast("integer")).alias("city"),
    F.lit("US").alias("country"),
    (F.col("id") % 62 + 18).cast("integer").alias("age"),
    F.date_sub(F.current_date(), (F.col("id") % 1095).cast("integer")).cast("timestamp").alias("created_at")
)
customers_df.writeTo(f"{CATALOG}.{NAMESPACE}.demo_customers").using("iceberg").createOrReplace()
print(f"Loaded {customers_df.count()} customers")

print("=== Loading demo_products (10,000 records) ===")
categories = F.array([F.lit(c) for c in ["Electronics","Clothing","Books","Home & Garden","Sports","Toys","Automotive","Food","Health","Beauty"]])

products_df = spark.range(1, 10001).select(
    F.concat(F.lit("PROD-"), F.lpad(F.col("id").cast("string"), 5, "0")).alias("product_id"),
    F.concat(F.lit("Product "), F.col("id").cast("string")).alias("product_name"),
    F.element_at(categories, (F.col("id") % 10 + 1).cast("integer")).alias("category"),
    F.round((F.col("id") % 995 + 5).cast("double"), 2).alias("price"),
    (F.col("id") % 500 + 10).cast("integer").alias("stock_quantity"),
    F.date_sub(F.current_date(), (F.col("id") % 730).cast("integer")).cast("timestamp").alias("created_at")
)
products_df.writeTo(f"{CATALOG}.{NAMESPACE}.demo_products").using("iceberg").createOrReplace()
print(f"Loaded {products_df.count()} products")

print("=== Loading demo_orders (200,000 records) ===")
statuses = F.array([F.lit(s) for s in ["completed","pending","shipped","cancelled","returned"]])

orders_df = spark.range(1, 200001).select(
    F.concat(F.lit("ORD-"), F.lpad(F.col("id").cast("string"), 7, "0")).alias("order_id"),
    F.concat(F.lit("CUST-"), F.lpad(((F.col("id") % 100000) + 1).cast("string"), 6, "0")).alias("customer_id"),
    F.concat(F.lit("PROD-"), F.lpad(((F.col("id") % 10000) + 1).cast("string"), 5, "0")).alias("product_id"),
    (F.col("id") % 5 + 1).cast("integer").alias("quantity"),
    F.round(((F.col("id") % 995) + 5).cast("double"), 2).alias("amount"),
    F.element_at(statuses, (F.col("id") % 5 + 1).cast("integer")).alias("status"),
    F.date_sub(F.current_date(), (F.col("id") % 365).cast("integer")).cast("timestamp").alias("order_date")
)
orders_df.writeTo(f"{CATALOG}.{NAMESPACE}.demo_orders").using("iceberg").createOrReplace()
print(f"Loaded {orders_df.count()} orders")

job.commit()
print("=== All tables loaded successfully! ===")
