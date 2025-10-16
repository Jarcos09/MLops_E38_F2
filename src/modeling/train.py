# src/train.py
import typer
import pandas as pd
from src.modeling.train_model import ModelTrainer
from src.config.config import conf

app = typer.Typer()

@app.command()
def train():
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
            "random_state": conf.training.random_state,
            "experiment_name": conf.training.experiment_name,
            "registry_model_name": conf.training.registry_model_name
        }
    )
    trainer.train_random_forest()
    trainer.train_xgboost()

if __name__ == "__main__":
    app()