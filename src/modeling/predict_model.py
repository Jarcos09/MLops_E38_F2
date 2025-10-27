import mlflow
import joblib
import pandas as pd
from loguru import logger

class ModelPredictor:
    def __init__(self, config):
        self.config = config
        self.model = None
        mlflow.set_tracking_uri(config.get("mlflow_tracking_uri", ""))

    def load_model(self):
        """
        Carga el modelo entrenado desde archivo local o desde el registro de MLflow.
        """
        model_type = self.config.get("use_model", "rf").lower()
        model_file = (
            self.config["rf_model_file"]
            if model_type == "rf"
            else self.config["xgb_model_file"]
        )

        logger.info(f"Cargando modelo '{model_type.upper()}' desde {model_file}")
        self.model = joblib.load(model_file)
        logger.success(f"Modelo {model_type.upper()} cargado correctamente.")

    def predict(self, X_new):
        """
        Genera predicciones con el modelo cargado.
        """
        if self.model is None:
            self.load_model()

        logger.info(f"Generando predicciones sobre {len(X_new)} muestras...")
        preds = self.model.predict(X_new)

        # Manejar salidas multioutput
        if preds.ndim == 1:
            preds_df = pd.DataFrame(preds, columns=["prediction"])
        else:
            preds_df = pd.DataFrame(
                preds, columns=[f"target_{i}" for i in range(preds.shape[1])]
            )

        logger.success("Predicciones generadas exitosamente.")
        return preds_df

    def save_predictions(self, preds_df):
        """
        Guarda las predicciones en disco y las registra en MLflow
        """
        preds_df.to_csv(self.config["output_file"], index=False)
        logger.info(f"Predicciones guardadas en {self.config["output_file"]}")

        # Configurar experimento de MLflow
        mlflow.set_experiment("Predictions_Tracking")

        # Registrar artefacto y tags
        with mlflow.start_run(run_name="Prediction_Run"):
            mlflow.log_artifact(str(self.config["output_file"]))
            mlflow.set_tags({
                "stage": "prediction",
                "model_type": self.config.get("use_model", "rf"),
                "data_source": self.config.get("data_source", "unknown"),
            })

    def run_prediction(self, X_new):
        """
        Flujo completo: carga modelo, predice, guarda y registra.
        """
        preds_df = self.predict(X_new)
        self.save_predictions(preds_df)
        return preds_df