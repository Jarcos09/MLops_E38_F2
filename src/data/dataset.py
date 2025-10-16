# src/dataset.py
import typer
from src.data.download_dataset import DatasetDownloader
from src.config.config import conf

app = typer.Typer()

@app.command()
def download():
    downloader = DatasetDownloader(
        dataset_id=conf.download.dataset_id,
        output_path=conf.data.raw_data_file
    )
    
    downloader.run()

if __name__ == "__main__":
    app()