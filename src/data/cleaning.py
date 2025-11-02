# cleaning.py
import typer
from src.data.clean_dataset import DatasetCleaner
from src.config.config import conf
from src.utils import paths

app = typer.Typer()

# Comando principal de la aplicación de limpieza de datasets
@app.command()
def main():
    # Asegura que exista la ruta de salida intermedia
    paths.ensure_path(conf.paths.interim)  # Crea la carpeta si no existe

    # Inicialización del objeto DatasetCleaner con rutas y umbral de asimetría
    cleaner = DatasetCleaner(
        input_path=conf.data.raw_data_file,         # Ruta del dataset original
        output_path=conf.data.interim_data_file,    # Ruta de salida para el dataset limpio
        skew_threshold=conf.cleaning.skew_threshold # Umbral de asimetría para imputación
    )
    
    # Ejecución del proceso completo de limpieza
    cleaner.run()  # Carga, limpia, convierte, imputa y guarda el dataset

# Ejecución del CLI cuando se ejecuta el script directamente
if __name__ == "__main__":
    app()  # Lanza la aplicación Typer