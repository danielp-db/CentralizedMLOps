# Databricks notebook source
# MAGIC %md # DESTROY ENVIRONMENT

# COMMAND ----------

# MAGIC %md ## Databases

# COMMAND ----------

from sklearn.datasets import fetch_california_housing

for environment in ["dev", "cert", "prod"]:
    try:
        spark.sql(f"DROP DATABASE {environment}_data CASCADE")
    except:
        print(f"Failed to drop {environment}_data")
    try:
        spark.sql(f"DROP DATABASE {environment}_fs CASCADE")
    except:
        print(f"Failed to drop {environment}_fs")

# COMMAND ----------

# MAGIC %md ## Data

# COMMAND ----------

dbutils.fs.rm("abfss://demo@danpstgacct1.dfs.core.windows.net/", True)