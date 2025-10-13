# -*- coding: utf-8 -*-

import sys 
import os
import numpy as np
import pandas as pd

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parents[2]))

from MLFlow.DVC.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR



FILENAME_INTERIM = "energy_modified_clean.csv"
FILENAME_PROCESSED = "energy_modified_processed.csv"

app = typer.Typer()




@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = INTERIM_DATA_DIR / FILENAME_INTERIM,
    output_path: Path = PROCESSED_DATA_DIR / FILENAME_PROCESSED,
    # -----------------------------------------
):

    logger.info(f"Cargando dataset limpio de {input_path}")

    test_size = 0.2
    random_state = 42

    df= pd.read_csv(input_path)

    # 1) Separar features y variables objetivo, eliminando columna de poco valor predictivo
    X = df.drop(columns=["Y1", "Y2", "mixed_type_col"])         # Features independientes
    y = df[["Y1", "Y2"]].copy()

    # 2) Convertir todas las columnas de X a tipo 'category'
    cat_features = X.columns.tolist()
    for col in cat_features:
        # Preparación para One-Hot Encoding
        X[col] = X[col].astype("category")


    # 3) Definir ColumnTransformer para OHE
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat",
             OneHotEncoder(drop="first", sparse_output=False, handle_unknown='ignore'),
             cat_features)
        ],
        remainder='drop'
    )

    # 4) Dividir en conjuntos de entrenamiento y prueba antes de OHE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


# 5) Transformación Yeo-Johnson para variables objetivo
    yao_transformer = PowerTransformer(method='yeo-johnson')
    yao_transformer.fit(y_train)                                # Ajustar transformador sobre entrenamiento
    y_train = pd.DataFrame(
        yao_transformer.transform(y_train),
        columns=["Y1", "Y2"],
        index=y_train.index
    )
    y_test = pd.DataFrame(
        yao_transformer.transform(y_test),
        columns=["Y1", "Y2"],
        index=y_test.index
    )

    # 6) Pipeline para OHE de X
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor)
    ])
    X_train_processed = full_pipeline.fit_transform(X_train)    # Ajuste y transformación en entrenamiento
    X_test_processed = full_pipeline.transform(X_test)
    
    print(X_train_processed.shape)
    print(X_test_processed.shape)
    print(y_train.shape)
    print(y_test.shape)

if __name__ == "__main__":
    app()
