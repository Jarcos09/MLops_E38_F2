from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from MLFlow.DVC.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR



FILENAME_INTERIM = "energy_modified_clean.csv"
FILENAME_PROCESSED = "energy_modified_processed.csv"

app = typer.Typer()




@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = FILENAME_INTERIM / FILENAME_INTERIM,
    output_path: Path = PROCESSED_DATA_DIR / FILENAME_PROCESSED,
    # -----------------------------------------
):

    print(input_path)


    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
