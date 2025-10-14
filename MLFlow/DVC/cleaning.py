# clean_dataset.py
import sys
import numpy as np
import pandas as pd
import typer
import yaml
from pathlib import Path
from scipy.stats import skew
from loguru import logger

# Ajuste de ruta para importar configuración institucional
sys.path.append(str(Path(__file__).resolve().parents[2]))
from MLFlow.DVC.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# Cargar parámetros desde params.yaml
def load_params(path: Path = Path(__file__).resolve().parents[2] / "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

@app.command()
def main():
    params = load_params()

    # Extraer parámetros
    raw_filename = params["cleaning"]["input_file"]
    cleaned_filename = params["cleaning"]["output_file"]
    strategy_threshold = params["cleaning"]["skew_threshold"]

    input_path = RAW_DATA_DIR / raw_filename
    output_path = INTERIM_DATA_DIR / cleaned_filename
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Cargando dataset original desde: {input_path}")
    df = pd.read_csv(input_path)

    logger.info("Reemplazando strings vacíos por NaN")
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    logger.info("Convirtiendo columnas a tipo numérico")
    df = df.apply(pd.to_numeric, errors='coerce')

    logger.info("Imputando valores faltantes según asimetría")
    for col in df.columns:
        col_skew = skew(df[col].dropna())
        if -strategy_threshold <= col_skew <= strategy_threshold:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].median())

    logger.success(f"Guardando dataset limpio en: {output_path}")
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    app()