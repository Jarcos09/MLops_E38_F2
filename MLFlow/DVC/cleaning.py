import sys 
import os
import numpy as np
import pandas as pd
import typer
from pathlib import Path
from scipy.stats import skew
from loguru import logger
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

# CLEANED_DATA_DIR = Path.home() / "MLops_E38_F2" / "data" / "cleaned"
from MLFlow.DVC.config import INTERIM_DATA_DIR, RAW_DATA_DIR 

app = typer.Typer()

# Nombre de los archivos locales
FILENAME_MODIFIED = "energy_modified.csv"
FILENAME_CLEANED = "energy_modified_clean.csv"

# Ruta donde se guardará el archivo (carpeta cleaned dentro de MLops_E38_F2 en el home)
INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)  # Crea la carpeta si no existe


@app.command()
def main():
    logger.info(f"Cargando dataset original de {RAW_DATA_DIR / FILENAME_MODIFIED}")
    
    # Cargar dataset original
    df_modified = pd.read_csv(RAW_DATA_DIR / FILENAME_MODIFIED)

    # Reemplazamos strings vacíos o con espacios por NaN
    logger.info("Reemplazando string")
    df_modified.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # Forzamos conversión de todas las columnas a numérico
    logger.info("Convertir a numéricos")
    df_modified = df_modified.apply(pd.to_numeric, errors='coerce')


    logger.info("Imputando datos faltantes")
    for col in df_modified.columns:
        # Calcular asimetría
        col_skew = skew(df_modified[col].dropna())

        # Decidir estrategia
        if -0.5 <= col_skew <= 0.5:
            # Aproximadamente simétrica → media
            df_modified[col] = df_modified[col].fillna(df_modified[col].mean())
        else:
            # Altamente sesgada → mediana
            df_modified[col] = df_modified[col].fillna(df_modified[col].median())

    logger.info(f"Guardando Dataset en {INTERIM_DATA_DIR / FILENAME_CLEANED}")
    df_modified.to_csv(INTERIM_DATA_DIR / FILENAME_CLEANED, index=False)


if __name__ == "__main__":
    app()
