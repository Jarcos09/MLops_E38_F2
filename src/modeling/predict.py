# src/predict.py
import typer
import pandas as pd
from src.modeling.predict_model import ModelPredictor
from src.config.config import conf
from src.utils import paths

app = typer.Typer()

# Comando principal para generar predicciones usando un modelo entrenado
@app.command()
def predict():
    """
    Carga un modelo entrenado y genera predicciones sobre un conjunto de datos.
    Guarda los resultados en un archivo CSV y los registra en MLflow si aplica.
    """
    # Asegura que exista la carpeta para guardar predicciones
    paths.ensure_path(conf.paths.prediction)    # Crea la carpeta si no existe

    # Carga del conjunto de datos de entrada (X_test)
    X_new = pd.read_csv(conf.data.processed_data.x_test_file)   # DataFrame de pruebas

    # Inicialización del objeto ModelPredictor con configuración
    predictor = ModelPredictor(
        config={
            "mlflow_tracking_uri": conf.training.mlflow_tracking_uri,   # URI de MLflow
            "rf_model_file": conf.training.rf_model_file,               # Ruta modelo RF
            "xgb_model_file": conf.training.xgb_model_file,             # Ruta modelo XGB
            "use_model": conf.prediction.use_model,                     # Modelo a usar
            "output_file": conf.data.prediction_file                    # Archivo de salida
        }
    )

    # Ejecución completa del flujo de predicción
    predictor.run_prediction(X_new)  # Carga modelo, predice, guarda y registra

# Ejecución del CLI cuando se ejecuta el script directamente
if __name__ == "__main__":
    app()  # Lanza la aplicación Typer