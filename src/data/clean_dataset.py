# clean_dataset.py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


from src.config.config import conf, PROJECT_PATHS, CLEANING_PATHS

class DatasetCleaner:
    def __init__(self, input_path: Path, output_path: Path, skew_threshold: float):
        self.input_path = input_path
        self.output_path = output_path
        self.skew_threshold = skew_threshold
        self.df = None

    def load_dataset(self):
        logger.info(f"Cargando dataset original desde: {self.input_path}")
        self.df = pd.read_csv(self.input_path)

    def replace_empty_strings(self):
        logger.info("Reemplazando strings vacíos por NaN")
        self.df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    def convert_to_numeric(self):
        logger.info("Convirtiendo columnas a tipo numérico")
        self.df = self.df.apply(pd.to_numeric, errors='coerce')

    def impute_missing_values(self):
        logger.info("Imputando valores faltantes según asimetría")
        for col in self.df.columns:
            col_skew = skew(self.df[col].dropna())
            if -self.skew_threshold <= col_skew <= self.skew_threshold:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            else:
                self.df[col] = self.df[col].fillna(self.df[col].median())

    def save_cleaned_dataset(self):
        logger.success(f"Guardando dataset limpio en: {self.output_path}")
        self.df.to_csv(self.output_path, index=False)

    def run(self):
        self.load_dataset()
        self.replace_empty_strings()
        self.convert_to_numeric()
        self.impute_missing_values()
        self.save_cleaned_dataset()