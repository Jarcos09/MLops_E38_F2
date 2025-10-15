# download_dataset.py
import gdown
from loguru import logger
import typer
from src.config.config import conf, PROJECT_PATHS, DOWNLOAD_PATHS

app = typer.Typer()

@app.command()
def main():
    PROJECT_PATHS.RAW.mkdir(parents=True, exist_ok=True)

    logger.info(f"Descargando dataset desde Google Drive (ID: {conf.download.dataset_id})...")
    gdown.download(id=conf.download.dataset_id, output=str(DOWNLOAD_PATHS.DATASET_FILE), quiet=True)
    logger.success(f"Archivo descargado exitosamente en: {DOWNLOAD_PATHS.DATASET_FILE}")

if __name__ == "__main__":
    app()