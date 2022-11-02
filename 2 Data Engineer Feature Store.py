# Databricks notebook source
dbutils.widgets.dropdown("centralized", "False", ["False", "True"])
isCentralized = dbutils.widgets.get("centralized") == "True"

dbutils.widgets.dropdown("environment", "dev", ["dev", "cert", "prod"])
environment = dbutils.widgets.get("environment")

# COMMAND ----------

# MAGIC %md # Connect to Feature Store

# COMMAND ----------

from databricks import feature_store

#if environment == "prod":
if isCentralized:
    scope = 'cmr_scope'
    prefix = 'cmr'

    feature_store_uri = f'databricks://{scope}:{prefix}'
    fs = feature_store.FeatureStoreClient(feature_store_uri=feature_store_uri)
else:
    fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %md # Transform Features

# COMMAND ----------

raw_data = spark.sql(f"SELECT * FROM {environment}_data.house_data")

# COMMAND ----------

import pyspark.sql.functions as F

log_columns = ["AveOccup", "AveBedrms", "AveRooms", "Population"]

transform_data = raw_data.select([F.log(c).alias(c) if c in log_columns else F.col(c) for c in raw_data.columns])

# COMMAND ----------

# MAGIC %md # Save Features

# COMMAND ----------

(
    transform_data.write
    #.option("path", f"abfss://demo@danpstgacct1.dfs.core.windows.net/{environment}_fs/data/house_features")
    .mode("overwrite")
    .saveAsTable(f"{environment}_fs.house_features")
)

# COMMAND ----------

# MAGIC %md # Register Feature Table

# COMMAND ----------

fs.register_table(delta_table = f"{environment}_fs.house_features",
                  primary_keys = "HouseId",
                  description = "Table containing data for house market")

# COMMAND ----------

# MAGIC %md # Promote all the way to Prod

# COMMAND ----------

# MAGIC %md Re-run this notebook with environment as prod

# COMMAND ----------

# MAGIC %md See the [Feature Table in the Feature Store](https://adb-6686503075854056.16.azuredatabricks.net/?o=6686503075854056#feature-store/feature-store)

# COMMAND ----------

# MAGIC %md #...[Back to the Data Scientist](https://adb-2173364778179441.1.azuredatabricks.net/?o=2173364778179441#notebook/4448106559485323/command/2159522006550863)
