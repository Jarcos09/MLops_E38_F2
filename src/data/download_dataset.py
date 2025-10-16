# src/data/download_dataset.py
import gdown
from loguru import logger
from pathlib import Path
from src.config.config import conf, PROJECT_PATHS, DOWNLOAD_PATHS

class DatasetDownloader:
    def __init__(self, dataset_id: str, output_path: Path):
        self.dataset_id = dataset_id
        self.output_path = output_path

    def prepare_directory(self):
        logger.debug(f"Creando directorio: {self.output_path.parent}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def download(self):
        logger.info(f"Descargando dataset desde Google Drive (ID: {self.dataset_id})...")
        gdown.download(id=self.dataset_id, output=str(self.output_path), quiet=True)
        logger.success(f"Archivo descargado exitosamente en: {self.output_path}")

    def run(self):
        self.prepare_directory()
        self.download()