# download_dataset.py
import sys
from pathlib import Path
import gdown
from loguru import logger
import typer
from config import conf, RAW_DATA_DIR, DOWNLOAD_DATASET_FILE

app = typer.Typer()

@app.command()
def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Descargando dataset desde Google Drive (ID: {conf.download.dataset_id})...")
    gdown.download(id=conf.download.dataset_id, output=str(DOWNLOAD_DATASET_FILE), quiet=True)
    logger.success(f"Archivo descargado exitosamente en: {DOWNLOAD_DATASET_FILE}")

if __name__ == "__main__":
    app()