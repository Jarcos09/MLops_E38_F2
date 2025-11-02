import pandas as pd
import typer
from src.modeling.plots_model import PlotGenerator
from src.config.config import conf
from loguru import logger
from src.utils import paths

app = typer.Typer()

# Comando principal para generar gráficos desde un dataset CSV
@app.command()
def main(
    input_path: str = conf.data.interim_data_file,
    plot_type: str = "histogram",
    column: str = None,
    x: str = None,
    y: str = None,
    filename: str = None
):
    """
    Genera diferentes tipos de gráficos desde un dataset CSV.
    Todos los gráficos se guardan en conf.paths.figures
    """
    # Carga del dataset
    logger.info(f"Cargando datos desde {input_path}")                       # Log de carga
    df = pd.read_csv(paths.ensure_path(input_path))                         # Lectura CSV
    plotter = PlotGenerator(df, paths.ensure_path(conf.paths.figures))      # Inicialización del generador de gráficos

    # Definición de nombre de archivo por defecto
    if filename is None:
        filename = f"{plot_type}.png"                                       # Nombre de archivo basado en el tipo de gráfico

    # Selección y generación del tipo de gráfico
    if plot_type == "histogram" and column:                                 # Histograma de una columna específica
        plotter.histogram(column=column, filename=filename)
    elif plot_type == "scatter" and x and y:                                # Gráfico de dispersión con columnas X e Y
        plotter.scatter(x=x, y=y, filename=filename)
    elif plot_type == "correlation":                                        # Matriz de correlación
        plotter.correlation_matrix(filename=filename)
    else:
        # Manejo de parámetros inválidos
        logger.error("Parámetros incorrectos para plot_type o columnas.")   # Error si parámetros no coinciden

# Ejecución del CLI cuando se ejecuta el script directamente
if __name__ == "__main__":
    app()  # Lanza la aplicación Typer