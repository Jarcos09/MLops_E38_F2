# preprocess_data.py
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import warnings
import yaml
from pathlib import Path
from loguru import logger
import typer

from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Configuración general
warnings.filterwarnings('ignore')
app = typer.Typer()

# Rutas institucionales
sys.path.append(str(Path(__file__).resolve().parents[2]))
from MLFlow.DVC.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Cargar parámetros desde params.yaml
def load_params(path: Path = Path(__file__).resolve().parents[2] / "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

@app.command()
def main():
    params = load_params()

    # Extraer rutas y nombres de archivo
    input_file = params["preprocessing"]["input_file"]
    output_files = params["preprocessing"]["output_files"]
    drop_columns = params["preprocessing"]["drop_columns"]
    target_columns = params["preprocessing"]["target_columns"]
    test_size = params["preprocessing"]["test_size"]
    random_state = params["preprocessing"]["random_state"]

    input_path = INTERIM_DATA_DIR / input_file
    output_Xtrain = PROCESSED_DATA_DIR / output_files["Xtrain"]
    output_Xtest = PROCESSED_DATA_DIR / output_files["Xtest"]
    output_ytrain = PROCESSED_DATA_DIR / output_files["ytrain"]
    output_ytest = PROCESSED_DATA_DIR / output_files["ytest"]

    logger.info(f"Cargando dataset limpio desde: {input_path}")
    df = pd.read_csv(input_path)

    # Separación de variables
    X = df.drop(columns=target_columns + drop_columns)
    y = df[target_columns].copy()

    # Conversión a categorías
    cat_features = X.columns.tolist()
    X[cat_features] = X[cat_features].astype("category")

    # Preprocesamiento con One-Hot Encoding
    encoder = OneHotEncoder(
        drop=params["preprocessing"]["encoding"]["drop"],
        sparse_output=params["preprocessing"]["encoding"]["sparse_output"],
        handle_unknown=params["preprocessing"]["encoding"]["handle_unknown"]
    )

    preprocessor = ColumnTransformer(
        transformers=[("cat", encoder, cat_features)],
        remainder='drop'
    )

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Transformación Yeo-Johnson
    yao = PowerTransformer(method=params["preprocessing"]["target_transform"])
    y_train = pd.DataFrame(yao.fit_transform(y_train), columns=y.columns, index=y_train.index)
    y_test = pd.DataFrame(yao.transform(y_test), columns=y.columns, index=y_test.index)

    # Pipeline de preprocesamiento
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc = pipeline.transform(X_test)

    # Guardado de datos procesados
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    pd.DataFrame(X_train_proc, columns=feature_names, index=X_train.index).to_csv(output_Xtrain, index=False)
    pd.DataFrame(X_test_proc, columns=feature_names, index=X_test.index).to_csv(output_Xtest, index=False)
    y_train.to_csv(output_ytrain, index=False)
    y_test.to_csv(output_ytest, index=False)

    logger.success("Preprocesamiento completado y archivos guardados.")

if __name__ == "__main__":
    app()