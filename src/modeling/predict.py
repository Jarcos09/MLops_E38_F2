from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import sys

from src.config.config import conf

app = typer.Typer()

@app.command()
def main(
    features_path: Path = Path(conf.paths.processed) / "test_features.csv",
    model_path: Path = Path(conf.paths.models) / "model.pkl",
    predictions_path: Path = Path(conf.paths.processed) / "test_predictions.csv",
):
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")

if __name__ == "__main__":
    app()
