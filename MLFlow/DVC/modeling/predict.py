from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import PROJECT_PATHS

app = typer.Typer()

@app.command()
def main(
    features_path: Path = PROJECT_PATHS.PROCESSED / "test_features.csv",
    model_path: Path = PROJECT_PATHS.MODELS / "model.pkl",
    predictions_path: Path = PROJECT_PATHS.PROCESSED / "test_predictions.csv",
):
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")

if __name__ == "__main__":
    app()
