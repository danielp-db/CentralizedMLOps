# Databricks notebook source
dbutils.widgets.dropdown("environment", "dev", ["dev", "cert", "prod"])
environment = dbutils.widgets.get("environment")

# COMMAND ----------

# MAGIC %md # EDA

# COMMAND ----------

df = spark.sql(f"SELECT * FROM {environment}_data.house_data")

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md # Model Training

# COMMAND ----------

# MAGIC %md ## Model 1

# COMMAND ----------

import pandas as pd

house_data = spark.sql(f"SELECT * FROM {environment}_data.house_data").toPandas()

X = house_data.set_index("HouseId").drop("MedHouseVal", axis=1).values
y = house_data.set_index("HouseId")["MedHouseVal"].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=182)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()

model.fit(X_train, y_train)
y_train_hat = model.predict(X_train)

mean_squared_error(y_train, y_train_hat)

# COMMAND ----------

# MAGIC %md ## Model 2

# COMMAND ----------

# MAGIC %md Log skewed variables

# COMMAND ----------

import mlflow
import pandas as pd
import numpy as np
from mlflow.models.signature import infer_signature

with mlflow.start_run() as run:
    house_data = spark.sql(f"SELECT * FROM {environment}_data.house_data").toPandas()

    log_columns = ["AveOccup", "AveBedrms", "AveRooms", "Population"]
    house_data[log_columns] = house_data[log_columns].apply(lambda x: np.log(x))

    X = house_data.set_index("HouseId").drop("MedHouseVal", axis=1).values
    y = house_data.set_index("HouseId")["MedHouseVal"].values

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

# COMMAND ----------

# MAGIC %md ## ...[Back to Data Engineer](https://adb-2173364778179441.1.azuredatabricks.net/?o=2173364778179441#notebook/4448106559485327)