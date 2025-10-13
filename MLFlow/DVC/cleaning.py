import sys 
import os
import numpy as np
import pandas as pd
import typer
from pathlib import Path
from loguru import logger
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

# CLEANED_DATA_DIR = Path.home() / "MLops_E38_F2" / "data" / "cleaned"
from MLFlow.DVC.config import CLEANED_DATA_DIR, RAW_DATA_DIR 

app = typer.Typer()

# Nombre de los archivos locales
FILENAME_MODIFIED = "energy_modified.csv"
FILENAME_CLEANED = "energy_modified_clean.csv"

# Ruta donde se guardará el archivo (carpeta cleaned dentro de MLops_E38_F2 en el home)
CLEANED_DATA_DIR = Path.home() / "MLops_E38_F2" / "data" / "cleaned"
CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)  # Crea la carpeta si no existe


@app.command()
def main():
    logger.info("Cargando dataset original")
    
    # Cargar dataset original
    df_modified = pd.read_csv(RAW_DATA_DIR / FILENAME_MODIFIED)
  
    # Reemplazamos strings vacíos o con espacios por NaN
    df_modified.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # Forzamos conversión de todas las columnas a numérico
    df_modified = df_modified.apply(pd.to_numeric, errors='coerce')

    print(df_modified)
  
if __name__ == "__main__":
    app()
