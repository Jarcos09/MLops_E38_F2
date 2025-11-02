from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# Clase modular para generar y guardar gráficos a partir de un DataFrame
class PlotGenerator:
    """
    Clase modular para generar gráficos a partir de DataFrames.
    Todos los gráficos se guardan en la ruta proporcionada.
    """

    # Inicialización de la clase con DataFrame y ruta de figuras
    def __init__(self, df: pd.DataFrame, figures_path: Path):
        self.df = df                                            # DataFrame con los datos
        self.figures_path = figures_path                        # Ruta donde se guardarán los gráficos
        self.figures_path.mkdir(parents=True, exist_ok=True)    # Crea la carpeta si no existe

    # Generación de histograma de una columna específica
    def histogram(self, column: str, filename: str, bins: int = 30, figsize=(8,6)):
        """Genera un histograma de una columna específica."""
        logger.info(f"Generando histograma para columna: {column}") # Log de inicio
        plt.figure(figsize=figsize)                                 # Configuración de tamaño de figura
        sns.histplot(self.df[column], bins=bins, kde=True)          # Histograma con KDE
        path = self.figures_path / filename                         # Ruta final del archivo
        plt.title(f'Histograma de {column}')                        # Título del gráfico
        plt.savefig(path)                                           # Guardado del gráfico
        plt.close()                                                 # Cierre de la figura para liberar memoria
        logger.info(f"Histograma guardado en: {path}")              # Log de éxito

    # Generación de scatter plot entre dos columnas
    def scatter(self, x: str, y: str, filename: str, hue: str = None, figsize=(8,6)):
        """Genera un scatter plot entre dos columnas."""
        logger.info(f"Generando scatter plot: {x} vs {y}")  # Log de inicio
        plt.figure(figsize=figsize)                         # Configuración de tamaño
        sns.scatterplot(data=self.df, x=x, y=y, hue=hue)    # Gráfico de dispersión
        path = self.figures_path / filename                 # Ruta final del archivo
        plt.title(f'Scatter Plot: {x} vs {y}')              # Título del gráfico
        plt.savefig(path)                                   # Guardado
        plt.close()