from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

from src.config.config import PROJECT_PATHS

app = typer.Typer()

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROJECT_PATHS.PROCESSED / "dataset.csv",
    output_path: Path = PROJECT_PATHS.FIGURES / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()