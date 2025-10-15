# clean_dataset.py
import numpy as np
import pandas as pd
import typer
from pathlib import Path
from scipy.stats import skew
from loguru import logger
from src.config.config  import conf, PROJECT_PATHS, CLEANING_PATHS

app = typer.Typer()

@app.command()
def main():
    PROJECT_PATHS.INTERIM.mkdir(parents=True, exist_ok=True)

    logger.info(f"Cargando dataset original desde: {CLEANING_PATHS.INPUT_FILE}")

    df = pd.read_csv(CLEANING_PATHS.INPUT_FILE)

    logger.info("Reemplazando strings vacíos por NaN")

    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    logger.info("Convirtiendo columnas a tipo numérico")

    df = df.apply(pd.to_numeric, errors='coerce')

    logger.info("Imputando valores faltantes según asimetría")

    for col in df.columns:
        col_skew = skew(df[col].dropna())
        if - conf.cleaning.skew_threshold <= col_skew <= conf.cleaning.skew_threshold:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].median())

    logger.success(f"Guardando dataset limpio en: {CLEANING_PATHS.OUTPUT_FILE}")
    df.to_csv(CLEANING_PATHS.OUTPUT_FILE, index=False)

if __name__ == "__main__":
    app()