# Databricks notebook source
dbutils.widgets.dropdown("environment", "dev", ["dev", "cert", "prod"])
environment = dbutils.widgets.get("environment")

dbutils.widgets.text("model_name", "")
model_name = dbutils.widgets.get("model_name")

dbutils.widgets.text("model_version", "")
model_version = dbutils.widgets.get("model_version")

dbutils.widgets.dropdown("output_environment", "dev", ["dev", "cert", "prod"])
output_environment = dbutils.widgets.get("output_environment")

# COMMAND ----------

# MAGIC %md # Connect to Feature Store

# COMMAND ----------

from databricks import feature_store

scope = 'cmr_scope'
prefix = 'cmr'

feature_store_uri = f'databricks://{scope}:{prefix}'
model_registry_uri = f'databricks://{scope}:{prefix}'
fs = feature_store.FeatureStoreClient(feature_store_uri=feature_store_uri, model_registry_uri=model_registry_uri)

# COMMAND ----------

# MAGIC %md # Connect to Model Tracking

# COMMAND ----------

import mlflow

model_tracking_uri = None #f'databricks://{scope}:{prefix}'
mlflow.set_tracking_uri(model_tracking_uri)

# COMMAND ----------

# MAGIC %md # Connect to Model Registry

# COMMAND ----------

model_registry_uri = f'databricks://{scope}:{prefix}'
mlflow.set_registry_uri(model_registry_uri)

# COMMAND ----------

# MAGIC %md # Model Consumption

# COMMAND ----------

# MAGIC %md ### Get Data From Feature Store

# COMMAND ----------

df = spark.sql(f"SELECT HouseId FROM {environment}_fs.house_features")

look_ups = [feature_store.FeatureLookup(
    table_name=f"{environment}_fs.house_features",
    lookup_key=["HouseId"],
)]

training_set = fs.create_training_set(df, feature_lookups=look_ups, exclude_columns=["HouseId","MedHouseVal"], label=None)#"MedHouseVal")
house_data = training_set.load_df()

# COMMAND ----------

# MAGIC %md ### Score Pandas Dataframe

# COMMAND ----------

model = mlflow.pyfunc.load_model(f'models:/{model_name}/{model_version}')

# COMMAND ----------

pandas_df = house_data.toPandas()

predictions = model.predict(pandas_df)

pandas_df["MedHouseVal_Predicted"] = predictions

# COMMAND ----------

pandas_df

# COMMAND ----------

# MAGIC %md ### Score Spark DataFrame

# COMMAND ----------

model_udf = mlflow.pyfunc.spark_udf(spark, f'models:/{model_name}/{model_version}')

# COMMAND ----------

spark_df = house_data

spark_df = spark_df.withColumn("MedHouseVal_Predicted", model_udf(*spark_df.columns))

# COMMAND ----------

display(spark_df)

# COMMAND ----------

# MAGIC %md ### Score With Feature Store

# COMMAND ----------

df = spark.sql(f"SELECT HouseId FROM {environment}_fs.house_features")
fs_df = fs.score_batch(f"models:/{model_name}_fs/{model_version}", df)

# COMMAND ----------

display(fs_df)

# COMMAND ----------

# MAGIC %md ### Save Predictions

# COMMAND ----------

(
    spark_df
    .write.mode("overwrite")
    .option("path", f"abfss://demo@danpstgacct1.dfs.core.windows.net/{output_environment}_data/data/house_preds")
    .saveAsTable(f"{output_environment}_data.house_preds")
)

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {output_environment}_data.house_preds"))