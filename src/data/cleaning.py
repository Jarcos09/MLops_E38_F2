# cleaning.py
import typer
from src.data.clean_dataset import DatasetCleaner
from src.config.config import conf
from src.utils import paths

app = typer.Typer()

@app.command()
def main():
    paths.ensure_path(conf.paths.interim)

    cleaner = DatasetCleaner(
        input_path=conf.data.raw_data_file,
        output_path=conf.data.interim_data_file,
        skew_threshold=conf.cleaning.skew_threshold
    )
    
    cleaner.run()

if __name__ == "__main__":
    app()
