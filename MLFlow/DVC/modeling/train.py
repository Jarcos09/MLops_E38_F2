from pathlib import Path

from loguru import logger
from tqdm import tqdm

import typer
import sys

# Rutas de configuración
sys.path.append(str(Path(__file__).resolve().parents[3]))

from MLFlow.DVC.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def load_params():
    with open("params.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

params = load_params()

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):

    logger.info(f"Cargando dataset original de {RAW_DATA_DIR / FILENAME_MODIFIED}")

if __name__ == "__main__":
    app()
