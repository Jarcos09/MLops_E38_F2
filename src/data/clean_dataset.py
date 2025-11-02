# clean_dataset.py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew
from loguru import logger
from pathlib import Path

## Definimos la CLASE DatasetCleaner
## input_path: ruta del archivo CSV de entrada.
## output_path: ruta donde se guardará el CSV limpio.
## skew_threshold: define un umbral para decidir si se usa la media o la mediana al imputar datos faltantes.
## self.df: contendrá el DataFrame cargado

class DatasetCleaner:
    # Inicialización de la clase y definición de parámetros principales
    def __init__(self, input_path: Path, output_path: Path, skew_threshold: float):
        self.input_path = input_path            # Ruta del dataset original
        self.output_path = output_path          # Ruta de salida del dataset limpio
        self.skew_threshold = skew_threshold    # Umbral de asimetría para decidir método de imputación
        self.df = None                          # DataFrame que contendrá los datos cargados

    # Carga del dataset original
    def load_dataset(self):
        logger.info(f"Cargando dataset original desde: {self.input_path}")  # Log informativo de carga
        self.df = pd.read_csv(self.input_path)                              # Lectura del archivo CSV

    # Reemplazo de cadenas vacías por valores nulos
    def replace_empty_strings(self):
        logger.info("Reemplazando strings vacíos por NaN")                  # Log de inicio de reemplazo
        self.df.replace(r'^\s*$', np.nan, regex=True, inplace=True)         # Sustituye espacios vacíos por NaN

    # Conversión de todas las columnas a tipo numérico
    def convert_to_numeric(self):
        logger.info("Convirtiendo columnas a tipo numérico")                # Log de conversión
        self.df = self.df.apply(pd.to_numeric, errors='coerce')             # Convierte columnas; valores no numéricos se vuelven NaN

    # Imputación de valores faltantes basada en la asimetría de cada columna
    def impute_missing_values(self):
        logger.info("Imputando valores faltantes según asimetría")          # Log de imputación
        for col in self.df.columns:                                         # Itera sobre todas las columnas
            col_skew = skew(self.df[col].dropna())                          # Calcula asimetría excluyendo NaN
            if -self.skew_threshold <= col_skew <= self.skew_threshold:     # Si la asimetría es baja
                self.df[col] = self.df[col].fillna(self.df[col].mean())     # Imputa con la media
            else:
                self.df[col] = self.df[col].fillna(self.df[col].median())   # Imputa con la mediana si la asimetría es alta

    # Guardado del dataset limpio
    def save_cleaned_dataset(self):
        logger.success(f"Guardando dataset limpio en: {self.output_path}")  # Log de guardado exitoso
        self.df.to_csv(self.output_path, index=False)                       # Exporta el DataFrame limpio a CSV

    # Ejecución completa del proceso de limpieza
    def run(self):
        self.load_dataset()             # Carga los datos
        self.replace_empty_strings()    # Limpia strings vacíos
        self.convert_to_numeric()       # Convierte a numérico
        self.impute_missing_values()    # Imputa valores faltantes
        self.save_cleaned_dataset()     # Guarda el dataset limpio