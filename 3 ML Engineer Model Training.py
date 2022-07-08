# Databricks notebook source
dbutils.widgets.dropdown("environment", "dev", ["dev", "cert", "prod"])
environment = dbutils.widgets.get("environment")

dbutils.widgets.text("model_name", "")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

# MAGIC %md # Model Training With Feature Store

# COMMAND ----------

# MAGIC %md ## Connect to Feature Store

# COMMAND ----------

from databricks import feature_store

scope = 'cmr_scope'
prefix = 'cmr'

feature_store_uri = f'databricks://{scope}:{prefix}'
model_registry_uri = f'databricks://{scope}:{prefix}'
fs = feature_store.FeatureStoreClient(feature_store_uri=feature_store_uri, model_registry_uri=model_registry_uri)

# COMMAND ----------

# MAGIC %md ## Connect to Model Tracking

# COMMAND ----------

import mlflow

model_tracking_uri = None #f'databricks://{scope}:{prefix}'
mlflow.set_tracking_uri(model_tracking_uri)

# COMMAND ----------

# MAGIC %md ## Connect to Model Registry

# COMMAND ----------

model_registry_uri = f'databricks://{scope}:{prefix}'
mlflow.set_registry_uri(model_registry_uri)

# COMMAND ----------

# MAGIC %md ## Get Data from Feature Store

# COMMAND ----------

df = spark.sql(f"SELECT HouseId, MedHouseVal FROM {environment}_fs.house_features")

look_ups = [feature_store.FeatureLookup(
    table_name=f"{environment}_fs.house_features",
    lookup_key=["HouseId"],
    feature_names=['AveBedrms', 'AveOccup', 'AveRooms', 'HouseAge', 'MedInc','Population']
)]

training_set = fs.create_training_set(df, feature_lookups=look_ups,
                                      exclude_columns=["HouseId"],
                                      label="MedHouseVal")#"MedHouseVal")
house_data = training_set.load_df()

# COMMAND ----------

# MAGIC %md ## Train Model

# COMMAND ----------

# MAGIC %md #### MLFlow Experiment

# COMMAND ----------

import mlflow
import pandas as pd
import numpy as np
from mlflow.models.signature import infer_signature

mlflow.end_run()
with mlflow.start_run() as run:
    pdf = house_data.toPandas()
    X = pdf.drop("MedHouseVal", axis=1)
    y = pdf["MedHouseVal"]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=182)

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    model = LinearRegression()

    model.fit(X_train, y_train)
    y_train_hat = model.predict(X_train)

    train_mse = mean_squared_error(y_train, y_train_hat)
    
    y_test_hat = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_hat)
    
    mlflow.log_params(model.get_params())
    mlflow.log_metrics({"train_mse": train_mse, "test_mse": test_mse})
    
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, artifact_path = "model", signature=signature)
    #mlflow.sklearn.log_model(model, artifact_path = "model")
    
    #If these metrics are satisfactory (Suggestion: automate hyperparameter tunning with hyperopt run together with looking at best model)
    run_id = run.info.run_id
mlflow.end_run()

# COMMAND ----------

# MAGIC %md #### MLFlow Model

# COMMAND ----------

# MAGIC %md ##### Without Feature Store

# COMMAND ----------

mlflow.register_model(model_uri=f'runs:/{run_id}/model', name=model_name)

# COMMAND ----------

# MAGIC %md ##### With Feature Store

# COMMAND ----------

fs.log_model(
  model,
  artifact_path="model",
  flavor=mlflow.sklearn,
  training_set=training_set,
  registered_model_name=f"{model_name}_fs"
)

# COMMAND ----------

# MAGIC %md Check the [model in the model registry](https://adb-6686503075854056.16.azuredatabricks.net/?o=6686503075854056#mlflow/models)