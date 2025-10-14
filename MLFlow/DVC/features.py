# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import warnings
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

# Rutas de configuración
sys.path.append(str(Path(__file__).resolve().parents[2]))
from MLFlow.DVC.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Archivos
FILENAME_INTERIM = "energy_modified_clean.csv"
FILENAME_Xtrain = "Xtrain_processed.csv"
FILENAME_Xtest = "Xtest_processed.csv"
FILENAME_ytrain = "ytrain_processed.csv"
FILENAME_ytest = "ytest_processed.csv"

@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / FILENAME_INTERIM,
    output_Xtrain: Path = PROCESSED_DATA_DIR / FILENAME_Xtrain,
    output_Xtest: Path = PROCESSED_DATA_DIR / FILENAME_Xtest,
    output_ytrain: Path = PROCESSED_DATA_DIR / FILENAME_ytrain,
    output_ytest: Path = PROCESSED_DATA_DIR / FILENAME_ytest,
):
    logger.info(f"Cargando dataset limpio desde: {input_path}")

    # Parámetros de partición
    test_size = 0.2
    random_state = 42

    # Carga de datos
    df = pd.read_csv(input_path)

    # Separación de variables
    X = df.drop(columns=["Y1", "Y2", "mixed_type_col"])
    y = df[["Y1", "Y2"]].copy()

    # Conversión a categorías
    cat_features = X.columns.tolist()
    X[cat_features] = X[cat_features].astype("category")

    # Preprocesamiento con One-Hot Encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown='ignore'), cat_features)
        ],
        remainder='drop'
    )

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Transformación Yeo-Johnson
    yao = PowerTransformer(method='yeo-johnson')
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
