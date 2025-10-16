# cleaning.py
import typer
import sys
from src.data.clean_dataset import DatasetCleaner
from src.config.config import conf, PROJECT_PATHS, CLEANING_PATHS

app = typer.Typer()

from loguru import logger

@app.command()
def main():
    PROJECT_PATHS.INTERIM.mkdir(parents=True, exist_ok=True)
    cleaner = DatasetCleaner(
        input_path=CLEANING_PATHS.INPUT_FILE,
        output_path=CLEANING_PATHS.OUTPUT_FILE,
        skew_threshold=conf.cleaning.skew_threshold
    )
    cleaner.run()

if __name__ == "__main__":
    app()
