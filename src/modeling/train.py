# src/train.py
import typer
import pandas as pd
from src.modeling.train_model import ModelTrainer
from src.config.config import conf
from src.utils import paths

app = typer.Typer()

@app.command()
def train():
    paths.ensure_path(conf.paths.models)

    X_train = pd.read_csv(conf.data.processed_data.x_train_file)
    X_test = pd.read_csv(conf.data.processed_data.x_test_file)
    y_train = pd.read_csv(conf.data.processed_data.y_train_file)
    y_test = pd.read_csv(conf.data.processed_data.y_test_file)

    trainer = ModelTrainer(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config={
            "mlflow_tracking_uri": conf.training.mlflow_tracking_uri,
            "random_state": conf.training.random_state,
            "rf_experiment_name": conf.training.rf_experiment_name,
            "rf_registry_model_name": conf.training.rf_registry_model_name,
            "rf_model_file": conf.training.rf_model_file,
            "xgb_experiment_name": conf.training.xgb_experiment_name,
            "xgb_registry_model_name": conf.training.xgb_registry_model_name,
            "xgb_model_file": conf.training.xgb_model_file
        }
    )
    trainer.train_random_forest()
    trainer.train_xgboost()

if __name__ == "__main__":
    app()