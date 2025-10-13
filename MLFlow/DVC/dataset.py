import sys 
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import gdown
from loguru import logger
from tqdm import tqdm
import typer

from MLFlow.DVC.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# IDs de los datasets en Google Drive
ID_DATASET_MODIFIED = "1OuJHPpn2Wv5EhlL1J98HWYyPJM-iL8Ds"

# Nombre del archivo local
FILENAME_MODIFIED = "energy_modified.csv"

# Ruta donde se guardará el archivo (carpeta RAW dentro de MLops_E38_F2 en el home)
RAW_DIR = Path.home() / "MLops_E38_F2" / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)  # Crea la carpeta si no existe

MODIFIED_PATH = RAW_DIR / FILENAME_MODIFIED


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    logger.info("Iniciando descarga del dataset desde Google Drive...")

    # Descarga del archivo
    gdown.download(id=ID_DATASET_MODIFIED, output=str(MODIFIED_PATH), quiet=True)

    logger.success(f"Archivo descargado exitosamente en: {MODIFIED_PATH}")

if __name__ == "__main__":
    app()
