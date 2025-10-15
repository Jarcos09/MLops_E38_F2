# clean_dataset.py
import sys
import numpy as np
import pandas as pd
import typer
from pathlib import Path
from scipy.stats import skew
from loguru import logger
from config  import conf, INTERIM_DATA_DIR, CLEAN_INPUT_PATH, CLEAN_OUTPUT_PATH

# Ajuste de ruta para importar configuración institucional
sys.path.append(str(Path(__file__).resolve().parents[2]))

app = typer.Typer()

@app.command()
def main():
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Cargando dataset original desde: {CLEAN_INPUT_PATH}")

    df = pd.read_csv(CLEAN_INPUT_PATH)

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

    logger.success(f"Guardando dataset limpio en: {CLEAN_OUTPUT_PATH}")
    df.to_csv(CLEAN_OUTPUT_PATH, index=False)

if __name__ == "__main__":
    app()