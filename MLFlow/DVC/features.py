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
from config import conf, PROJECT_PATHS, PREPROCESSING_PATHS

from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Configuración general
warnings.filterwarnings('ignore')
app = typer.Typer()

PROJECT_PATHS.PROCESSED.mkdir(parents=True, exist_ok=True)

@app.command()
def main():
    logger.info(f"Cargando dataset limpio desde: {PREPROCESSING_PATHS.INPUT_FILE}")
    df = pd.read_csv(PREPROCESSING_PATHS.INPUT_FILE)

    # Separación de variables
    X = df.drop(columns=conf.preprocessing.target_columns + conf.preprocessing.drop_columns)
    y = df[conf.preprocessing.target_columns].copy()

    # Conversión a categorías
    cat_features = X.columns.tolist()
    X[cat_features] = X[cat_features].astype("category")

    # Preprocesamiento con One-Hot Encoding
    encoder = OneHotEncoder(
        drop=conf.preprocessing.encoding.drop,
        sparse_output=conf.preprocessing.encoding.sparse_output,
        handle_unknown=conf.preprocessing.encoding.handle_unknown
    )

    preprocessor = ColumnTransformer(
        transformers=[("cat", encoder, cat_features)],
        remainder='drop'
    )

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=conf.preprocessing.test_size, random_state=conf.preprocessing.random_state)

    # Transformación Yeo-Johnson
    yao = PowerTransformer(method=conf.preprocessing.target_transform)
    y_train = pd.DataFrame(yao.fit_transform(y_train), columns=y.columns, index=y_train.index)
    y_test = pd.DataFrame(yao.transform(y_test), columns=y.columns, index=y_test.index)

    # Pipeline de preprocesamiento
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc = pipeline.transform(X_test)

    # Guardado de datos procesados
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    pd.DataFrame(X_train_proc, columns=feature_names, index=X_train.index).to_csv(PREPROCESSING_PATHS.X_TRAIN, index=False)
    pd.DataFrame(X_test_proc, columns=feature_names, index=X_test.index).to_csv(PREPROCESSING_PATHS.X_TEST, index=False)
    y_train.to_csv(PREPROCESSING_PATHS.Y_TRAIN, index=False)
    y_test.to_csv(PREPROCESSING_PATHS.Y_TEST, index=False)

    logger.success("Preprocesamiento completado y archivos guardados.")

if __name__ == "__main__":
    app()