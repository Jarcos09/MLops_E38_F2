# src/modeling/train_model.py
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
from src.config.config import conf

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test, config):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.config = config
        self.best_rf_model = None

    def train_random_forest(self):
        logger.info("Iniciando búsqueda de hiperparámetros para Random Forest Multi-Output")
        rf_base = RandomForestRegressor(random_state=self.config["random_state"])
        multioutput_rf = MultiOutputRegressor(rf_base)

        param_grid = self.config.get("rf_param_grid", {
            "estimator__n_estimators": [100, 200],
            "estimator__max_depth": [8, 12, None],
            "estimator__min_samples_split": [5, 10]
        })

        grid_reg = GridSearchCV(
            multioutput_rf,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0
        )
        grid_reg.fit(self.X_train, self.y_train.values)
        self.best_rf_model = grid_reg.best_estimator_

        logger.success("Random Forest entrenado con éxito.")
        return self.best_rf_model

    def train_xgboost_with_mlflow(self):
        rf_params = self.best_rf_model.estimator.get_params()
        logger.info("Inicializando modelo XGBoost Multi-Output con hiperparámetros heredados de Random Forest")

        xgb_base = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=rf_params.get('n_estimators', 100),
            learning_rate=0.05,
            max_depth=rf_params.get('max_depth', 6),
            random_state=rf_params.get('random_state', self.config["random_state"]),
            n_jobs=-1
        )
        multioutput_model = MultiOutputRegressor(xgb_base)

        mlflow.set_experiment(self.config["experiment_name"])
        with mlflow.start_run() as run:
            logger.info("Entrenando modelo XGBoost Multi-Output...")
            multioutput_model.fit(self.X_train, self.y_train)
            logger.success("Entrenamiento completado.")

            y_pred = multioutput_model.predict(self.X_test)
            self.log_metrics(y_pred)
            self.log_model(multioutput_model, run.info.run_id)

    def log_metrics(self, y_pred):
        rmse_y1 = np.sqrt(mean_squared_error(self.y_test["Y1"], y_pred[:, 0]))
        rmse_y2 = np.sqrt(mean_squared_error(self.y_test["Y2"], y_pred[:, 1]))
        r2_y1 = r2_score(self.y_test["Y1"], y_pred[:, 0])
        r2_y2 = r2_score(self.y_test["Y2"], y_pred[:, 1])

        logger.info("--- Resultados de Evaluación (XGBoost) ---")
        logger.info(f"Y1 → RMSE: {rmse_y1:.4f}, R²: {r2_y1:.4f}")
        logger.info(f"Y2 → RMSE: {rmse_y2:.4f}, R²: {r2_y2:.4f}")

        mlflow.log_metrics({
            "rmse_y1": rmse_y1,
            "rmse_y2": rmse_y2,
            "r2_y1": r2_y1,
            "r2_y2": r2_y2
        })

    def log_model(self, model, run_id):
        input_example = self.X_train.iloc[:2]
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="xgb_multioutput_model",
            input_example=input_example
        )
        model_uri = f"runs:/{run_id}/xgb_multioutput_model"
        mlflow.register_model(model_uri=model_uri, name=self.config["registry_model_name"])
        logger.success(f"Modelo registrado como '{self.config['registry_model_name']}'")