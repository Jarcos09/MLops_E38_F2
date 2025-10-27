# src/predict.py
import typer
import pandas as pd
from src.modeling.predict_model import ModelPredictor
from src.config.config import conf
from src.utils import paths

app = typer.Typer()

@app.command()
def predict():
    """
    Carga un modelo entrenado y genera predicciones sobre un conjunto de datos.
    Guarda los resultados en un archivo CSV y los registra en MLflow si aplica.
    """
    # Asegurar rutas
    paths.ensure_path(conf.paths.prediction)

    # Cargar datos de entrada
    X_new = pd.read_csv(conf.data.processed_data.x_test_file)

    predictor = ModelPredictor(
        config={
            "mlflow_tracking_uri": conf.training.mlflow_tracking_uri,
            "rf_model_file": conf.training.rf_model_file,
            "xgb_model_file": conf.training.xgb_model_file,
            "use_model": conf.prediction.use_model,
            "output_file": conf.data.prediction_file
        }
    )

    predictor.run_prediction(X_new)

if __name__ == "__main__":
    app()