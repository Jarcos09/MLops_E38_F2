import mlflow
import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor, MultiOutputRegressor
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class ModelTrainer:
    ## El constructor de la clase inicializa el estado del objeto ModelTrainer. 
    ## Recibe los datos preparados y un diccionario de configuración, 
    ## y establece la conexión con el servidor de tracking de MLFlow.
    ## PARAMETROS
    # X_train (DataFrame/Array): Características para el conjunto de entrenamiento.
    # X_test (DataFrame/Array): Características para el conjunto de prueba.
    # y_train (Series/Array): Etiquetas/objetivos para el conjunto de entrenamiento.
    # y_test (Series/Array): Etiquetas/objetivos para el conjunto de prueba.
    # config (dict): Un diccionario que contiene los parámetros de configuración del proyecto (leído de params.yaml).
    
    def __init__(self, X_train, X_test, y_train, y_test, config):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.config = config
        self.input_example = self.X_train.iloc[:2]
        mlflow.set_tracking_uri(self.config.get("mlflow_tracking_uri", ""))

    def train_random_forest(self):
        """
        Entrena un modelo Random Forest Multi-Output con búsqueda manual de hiperparámetros.
        Registra cada combinación como run anidado en MLflow y selecciona el mejor modelo
        según la métrica definida en config["best_metric"].
        """
        logger.info("Iniciando búsqueda de hiperparámetros para Random Forest Multi-Output")

        # Configuración del grid y de la métrica a optimizar
        param_grid = self.config.get("rf_param_grid", {
            "estimator__n_estimators": [100, 200],
            "estimator__max_depth": [8, 12, None],
            "estimator__min_samples_split": [5, 10]
        })
        best_metric_name = self.config.get("best_metric", "rmse").lower()

        mlflow.set_experiment(self.config["rf_experiment_name"])

        with mlflow.start_run(run_name="RandomForest_Tuning") as parent_run:
            best_score = np.inf if best_metric_name in ["rmse", "mse", "mae"] else -np.inf
            best_params = None
            best_model = None

            logger.info(f"Usando métrica objetivo: {best_metric_name.upper()}")

            for i, params in enumerate(ParameterGrid(param_grid)):
                with mlflow.start_run(run_name=f"trial_{i}", nested=True) as child_run:
                    logger.info(f"Trial {i}: {params}")

                    rf = RandomForestRegressor(
                        random_state=self.config["random_state"],
                        **{k.replace("estimator__", ""): v for k, v in params.items()}
                    )
                    model = MultiOutputRegressor(rf)
                    model.fit(self.X_train, self.y_train)

                    # Predicción y evaluación
                    y_pred = model.predict(self.X_test)
                    metrics = {
                        "mse": mean_squared_error(self.y_test, y_pred),
                        "rmse": np.sqrt(mean_squared_error(self.y_test, y_pred)),
                        "mae": mean_absolute_error(self.y_test, y_pred),
                        "r2": r2_score(self.y_test, y_pred)
                    }

                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        name=self.config["rf_registry_model_name"],
                        input_example=self.input_example
                    )

                    # Selección del mejor modelo
                    metric_value = metrics[best_metric_name]
                    if (
                        best_metric_name in ["rmse", "mse", "mae"] and metric_value < best_score
                    ) or (
                        best_metric_name in ["r2"] and metric_value > best_score
                    ):
                        best_score = metric_value
                        best_params = params
                        best_model = model

            logger.success(f"Mejor modelo guardado con {best_metric_name.upper()} = {best_score:.4f}")
            self.best_rf_model = best_model

            # Log métricas finales detalladas y registro del modelo
            y_pred_best = best_model.predict(self.X_test)
            final_metrics = self.log_metrics(y_pred_best)
            self.log_model(
                best_model,
                self.config["rf_registry_model_name"], 
                self.config["rf_model_file"], 
                params=best_params
            )

            return best_model, final_metrics

    def train_xgboost(self):
        """
        Entrena un modelo XGBoost Multi-Output con búsqueda opcional de hiperparámetros.
        Registra métricas y el mejor modelo en MLflow.
        """

        logger.info("Iniciando entrenamiento de modelo XGBoost Multi-Output")

        # Configuración base
        base_params = {
            "objective": "reg:squarederror",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "random_state": self.config.get("random_state", 42),
            "n_jobs": -1,
        }

        # Grid de hiperparámetros (si existe en config)
        param_grid = self.config.get("xgb_param_grid", {
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [4, 6, 8],
            "n_estimators": [200, 400],
            "subsample": [0.8, 1.0]
        })
        best_metric_name = self.config.get("best_metric", "rmse").lower()

        mlflow.set_experiment(self.config["xgb_experiment_name"])

        with mlflow.start_run(run_name="XGBoost_Tuning") as parent_run:
            best_score = np.inf if best_metric_name in ["rmse", "mse", "mae"] else -np.inf
            best_params = None
            best_model = None

            logger.info(f"Usando métrica de optimización: {best_metric_name.upper()}")

            for i, params in enumerate(ParameterGrid(param_grid)):
                full_params = {**base_params, **params}

                with mlflow.start_run(run_name=f"xgb_trial_{i}", nested=True):
                    logger.info(f"Trial {i}: {full_params}")

                    # Entrenamiento
                    xgb = XGBRegressor(**full_params)
                    model = MultiOutputRegressor(xgb)
                    model.fit(self.X_train, self.y_train)

                    # Predicción y métricas
                    y_pred = model.predict(self.X_test)
                    mse = np.mean((self.y_test - y_pred) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(self.y_test - y_pred))

                    metrics = {"mse": mse, "rmse": rmse, "mae": mae}
                    mlflow.log_params(full_params)
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        name=self.config["xgb_registry_model_name"],
                        input_example=self.input_example
                    )

                    # Selección del mejor modelo
                    metric_value = metrics[best_metric_name]
                    if (
                        best_metric_name in ["rmse", "mse", "mae"] and metric_value < best_score
                    ) or (
                        best_metric_name in ["r2"] and metric_value > best_score
                    ):
                        best_score = metric_value
                        best_params = full_params
                        best_model = model

            # Registrar mejor resultado
            logger.success(f"Mejor modelo encontrado con {best_metric_name.upper()} = {best_score:.4f}")

            # Log métricas finales detalladas y registro del modelo
            y_pred_best = best_model.predict(self.X_test)
            final_metrics = self.log_metrics(y_pred_best)
            self.log_model(
                best_model, 
                self.config["xgb_registry_model_name"], 
                self.config["xgb_model_file"], 
                params=best_params
            )

            return best_model, final_metrics

    def log_metrics(self, y_pred):
            """
            Registra métricas (RMSE y R²) para cada variable de salida en MLflow.
            Admite múltiples salidas (MultiOutputRegressor).
            """
            logger.info("Calculando métricas por variable de salida...")

            metrics = {}
            num_outputs = y_pred.shape[1]

            for i in range(num_outputs):
                target_name = self.y_test.columns[i]
                rmse = np.sqrt(mean_squared_error(self.y_test[target_name], y_pred[:, i]))
                r2 = r2_score(self.y_test[target_name], y_pred[:, i])

                metrics[f"rmse_{target_name}"] = rmse
                metrics[f"r2_{target_name}"] = r2

                logger.info(f"{target_name} -> RMSE: {rmse:.4f}, R²: {r2:.4f}")

            # Métricas promedio
            avg_rmse = np.mean([v for k, v in metrics.items() if k.startswith("rmse_")])
            avg_r2 = np.mean([v for k, v in metrics.items() if k.startswith("r2_")])
            metrics["avg_rmse"] = avg_rmse
            metrics["avg_r2"] = avg_r2

            logger.info(f"Promedio -> RMSE: {avg_rmse:.4f}, R²: {avg_r2:.4f}")

            # Log a MLflow
            mlflow.log_metrics(metrics)

            return metrics

    def log_model(self, model, model_name, model_path, params=None):
            """
            Registra un modelo en MLflow con ejemplo de entrada, parámetros y versión.
            Si el modelo ya existe, crea una nueva versión.
            """
            logger.info(f"Registrando modelo '{model_name}' en MLflow...")

            # Guardar modelo localmente
            joblib.dump(model, model_path)

            # Loggear hiperparámetros si se pasan
            if params is not None:
                logger.info("Registrando hiperparámetros del mejor modelo...")
                mlflow.log_params(params)
            else:
                # Si no se pasa un dict explícito, intentar extraer del estimador base
                try:
                    model_params = model.estimators_[0].get_params() if hasattr(model, "estimators_") else model.get_params()
                    mlflow.log_params(model_params)
                except Exception as e:
                    logger.warning(f"No se pudieron registrar los parámetros: {e}")

            # Log del modelo en MLflow
            mlflow.sklearn.log_model(
                sk_model=model,
                name=model_name,
                input_example=self.input_example,
                registered_model_name=model_name
            )

            # Log del archivo serializado localmente
            mlflow.log_artifact(model_path)

            # Etiquetas informativas
            mlflow.set_tag("model_stage", "tuned_best")
            mlflow.set_tag("model_type", type(model).__name__)
            mlflow.set_tag("framework", "scikit-learn")
            mlflow.set_tag("multioutput", True)

            logger.success(f"Modelo '{model_name}' registrado y versionado con éxito en MLflow.")
