# Databricks notebook source
# MAGIC %md # CREATE ENVIRONMENT CAMBIO

# COMMAND ----------

Adds cell

# COMMAND ----------

from sklearn.datasets import fetch_california_housing

for environment in ["dev", "cert", "prod"]:
    spark.sql(f"""CREATE DATABASE IF NOT EXISTS {environment}_data""")
                #LOCATION 'abfss://demo@danpstgacct1.dfs.core.windows.net/{environment}_data/db'""")
    spark.sql(f"""CREATE DATABASE IF NOT EXISTS {environment}_fs""")
                #LOCATION 'abfss://demo@danpstgacct1.dfs.core.windows.net/{environment}_fs/db'""")

    X, y = fetch_california_housing(return_X_y=True, as_frame=True)

    df = X.copy()
    df[y.name] = y
    df = df.rename_axis('HouseId').reset_index()

    house_data = df.drop(["Latitude", "Longitude"], axis=1)
    house_locations = df[["HouseId", "Latitude", "Longitude"]]

    (
        spark.createDataFrame(house_data)
        .write.mode("overwrite")
        #.option("path", f"abfss://demo@danpstgacct1.dfs.core.windows.net/{environment}_data/data/house_data")
        #.option("path", f"abfss://demo@danpstgacct1.dfs.core.windows.net/{environment}_data/data/house_data")
        .saveAsTable(f"{environment}_data.house_data")
    )
    (
        spark.createDataFrame(house_locations)
        .write.mode("overwrite")
        #.option("path", f"abfss://demo@danpstgacct1.dfs.core.windows.net/{environment}_data/data/house_locations")
        .saveAsTable(f"{environment}_data.house_locations")
    )
