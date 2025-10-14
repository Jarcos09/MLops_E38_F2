# download_dataset.py
import sys
from pathlib import Path
import yaml
import gdown
from loguru import logger
import typer

# Ajuste de ruta para importar configuración
sys.path.append(str(Path(__file__).resolve().parents[2]))
from MLFlow.DVC.config import RAW_DATA_DIR

app = typer.Typer()

# Cargar parámetros desde params.yaml
def load_params(path: Path = Path(__file__).resolve().parents[2] / "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@app.command()
def main():
    params = load_params()
    dataset_id = params["download"]["dataset_id"]
    filename = params["download"]["filename"]
    output_path = RAW_DATA_DIR / filename

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Descargando dataset desde Google Drive (ID: {dataset_id})...")
    gdown.download(id=dataset_id, output=str(output_path), quiet=True)
    logger.success(f"Archivo descargado exitosamente en: {output_path}")

if __name__ == "__main__":
    app()