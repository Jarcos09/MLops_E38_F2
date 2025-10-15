# train_model.py
from pathlib import Path
from loguru import logger
import typer
import yaml
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from loguru import logger
import mlflow

# Rutas de configuración institucional
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import conf, MODELS_DIR, PREPROCESSING_OUTPUT_XTRAIN, PREPROCESSING_OUTPUT_XTEST, PREPROCESSING_OUTPUT_YTRAIN, PREPROCESSING_OUTPUT_YTEST

app = typer.Typer()


def train_and_evaluate_rf(X_train, X_test, y_train, y_test, random_state=42, param_grid=None, metrics_path="metrics.txt"):

    rf_base = RandomForestRegressor(random_state=random_state)
    multioutput_rf = MultiOutputRegressor(rf_base)

    if param_grid is None:
        param_grid = {
            "estimator__n_estimators": [100, 200],
            "estimator__max_depth": [8, 12, None],
            "estimator__min_samples_split": [5, 10]
        }

    logger.info("Iniciando búsqueda de hiperparámetros para Random Forest Multi-Output")

    grid_reg = GridSearchCV(
        multioutput_rf,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=0
    )

    grid_reg.fit(X_train, y_train.values)
    best_rf_reg = grid_reg.best_estimator_
    y_pred = best_rf_reg.predict(X_test)

    # Métricas
    rmse_y1 = np.sqrt(mean_squared_error(y_test["Y1"], y_pred[:, 0]))
    rmse_y2 = np.sqrt(mean_squared_error(y_test["Y2"], y_pred[:, 1]))
    r2_y1 = r2_score(y_test["Y1"], y_pred[:, 0])
    r2_y2 = r2_score(y_test["Y2"], y_pred[:, 1])

    # Guardar métricas
    #with open(metrics_path, "w") as f:
    #    f.write("=== Random Forest Regressor Multi-Output ===\n")
    #    f.write(f"Mejores parámetros: {grid_reg.best_params_}\n")
    #    f.write("--- Resultados de Evaluación ---\n")
    #    f.write(f"Métricas para Y1:\n  RMSE: {rmse_y1:.4f}\n  R^2 Score: {r2_y1:.4f}\n")
    #    f.write(f"Métricas para Y2:\n  RMSE: {rmse_y2:.4f}\n  R^2 Score: {r2_y2:.4f}\n")

    #logger.success("Entrenamiento completado. Métricas guardadas en metrics.txt")
    return best_rf_reg

import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from loguru import logger

def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    best_rf_reg,
    random_state=42,
    metrics_path="metrics_xgb.txt",
    experiment_name="xgb_multioutput",
    registry_model_name="XGBMultiOutputJuan"
):
    rf_params = best_rf_reg.estimator.get_params()

    logger.info("Inicializando modelo XGBoost Multi-Output con hiperparámetros heredados de Random Forest")

    xgb_base = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=rf_params.get('n_estimators', 100),
        learning_rate=0.05,
        max_depth=rf_params.get('max_depth', 6),
        random_state=rf_params.get('random_state', random_state),
        n_jobs=-1
    )

    multioutput_model = MultiOutputRegressor(xgb_base)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        logger.info("Entrenando modelo XGBoost Multi-Output...")
        multioutput_model.fit(X_train, y_train)
        logger.success("Entrenamiento completado.")

        y_pred_test = multioutput_model.predict(X_test)

        # Métricas
        mse_y1 = mean_squared_error(y_test["Y1"], y_pred_test[:, 0])
        mse_y2 = mean_squared_error(y_test["Y2"], y_pred_test[:, 1])
        rmse_y1 = np.sqrt(mse_y1)
        rmse_y2 = np.sqrt(mse_y2)
        r2_y1 = r2_score(y_test["Y1"], y_pred_test[:, 0])
        r2_y2 = r2_score(y_test["Y2"], y_pred_test[:, 1])

        logger.info("--- Resultados de Evaluación (XGBoost) ---")
        logger.info(f"Número de features: {X_train.shape[1]}")
        logger.info(f"Tamaño del set de prueba: {X_test.shape[0]} observaciones")
        logger.info(f"Y1 → RMSE: {rmse_y1:.4f}, R²: {r2_y1:.4f}")
        logger.info(f"Y2 → RMSE: {rmse_y2:.4f}, R²: {r2_y2:.4f}")

        # Log de hiperparámetros
        mlflow.log_params({
            "n_estimators": xgb_base.get_params()["n_estimators"],
            "max_depth": xgb_base.get_params()["max_depth"],
            "learning_rate": xgb_base.get_params()["learning_rate"],
            "random_state": random_state
        })

        # Log de métricas
        mlflow.log_metrics({
            "rmse_y1": rmse_y1,
            "rmse_y2": rmse_y2,
            "r2_y1": r2_y1,
            "r2_y2": r2_y2
        })

        # Ejemplo de entrada para inferencia de firma
        input_example = X_train.iloc[:2]

        # Log del modelo
        mlflow.sklearn.log_model(
            sk_model=multioutput_model,
            artifact_path="xgb_multioutput_model",
            input_example=input_example
        )

        # Registro en el Model Registry
        model_uri = f"runs:/{run.info.run_id}/xgb_multioutput_model"
        mlflow.register_model(model_uri=model_uri, name=registry_model_name)
        logger.success(f"Modelo registrado en el MLflow Model Registry como '{registry_model_name}'")

        # Guardar métricas localmente
        #with open(metrics_path, "w") as f:
        #    f.write("=== XGBoost Regressor Multi-Output ===\n")
        #    f.write("--- Resultados de Evaluación ---\n")
        #    f.write(f"Número de features: {X_train.shape[1]}\n")
        #    f.write(f"Tamaño del set de prueba: {X_test.shape[0]} observaciones\n")
        #    f.write(f"Métricas para Y1:\n  RMSE: {rmse_y1:.4f}\n  R^2 Score: {r2_y1:.4f}\n")
        #    f.write(f"Métricas para Y2:\n  RMSE: {rmse_y2:.4f}\n  R^2 Score: {r2_y2:.4f}\n")

    #return multioutput_model


@app.command()
def main(
    Xtrain_path: Path =PREPROCESSING_OUTPUT_XTRAIN,
    Xtest_path: Path = PREPROCESSING_OUTPUT_XTEST,
    ytrain_path: Path = PREPROCESSING_OUTPUT_YTRAIN,
    ytest_path: Path = PREPROCESSING_OUTPUT_YTEST,
    model_path: Path = MODELS_DIR / conf.training.model_filename
):
    logger.info(f"Cargando características de entrenamiento desde: {Xtrain_path}")
    logger.info(f"Cargando características de prueba desde: {Xtest_path}")
    logger.info(f"Cargando etiquetas de entrenamiento desde: {ytrain_path}")
    logger.info(f"Cargando etiquetas de prueba desde: {ytest_path}")
    logger.info(f"Ruta destino para el modelo entrenado: {model_path}")

    X_train = pd.read_csv(Xtrain_path)
    X_test = pd.read_csv(Xtest_path)
    y_train = pd.read_csv(ytrain_path)
    y_test = pd.read_csv(ytest_path)


    best_model = train_and_evaluate_rf(X_train, X_test, y_train, y_test, random_state=conf.training.random_state)
    train_model(X_train, X_test, y_train, y_test, best_model, random_state=conf.training.random_state)


if __name__ == "__main__":
    app()