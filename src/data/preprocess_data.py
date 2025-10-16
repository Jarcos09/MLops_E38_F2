# src/data/preprocess_data.py
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, input_path: Path, output_paths: dict, config: dict):
        self.input_path = input_path
        self.output_paths = output_paths
        self.config = config
        self.df = None
        self.X = None
        self.y = None
        self.pipeline = None
        self.feature_names = None

    def load_data(self):
        logger.info(f"Cargando dataset limpio desde: {self.input_path}")
        self.df = pd.read_csv(self.input_path)

    def separate_variables(self):
        self.X = self.df.drop(columns=self.config["target_columns"] + self.config["drop_columns"])
        self.y = self.df[self.config["target_columns"]].copy()

    def encode_features(self):
        cat_features = self.X.columns.tolist()
        self.X[cat_features] = self.X[cat_features].astype("category")

        encoder = OneHotEncoder(
            drop=self.config["encoding"]["drop"],
            sparse_output=self.config["encoding"]["sparse_output"],
            handle_unknown=self.config["encoding"]["handle_unknown"]
        )

        self.pipeline = Pipeline(steps=[
            ("preprocessor", ColumnTransformer(
                transformers=[("cat", encoder, cat_features)],
                remainder='drop'
            ))
        ])

    def split_and_transform(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.config["test_size"],
            random_state=self.config["random_state"]
        )

        yao = PowerTransformer(method=self.config["target_transform"])
        y_train = pd.DataFrame(yao.fit_transform(y_train), columns=self.y.columns, index=y_train.index)
        y_test = pd.DataFrame(yao.transform(y_test), columns=self.y.columns, index=y_test.index)

        X_train_proc = self.pipeline.fit_transform(X_train)
        X_test_proc = self.pipeline.transform(X_test)
        self.feature_names = self.pipeline.named_steps['preprocessor'].get_feature_names_out()

        return X_train_proc, X_test_proc, y_train, y_test, X_train.index, X_test.index

    def save_outputs(self, X_train_proc, X_test_proc, y_train, y_test, train_idx, test_idx):
        pd.DataFrame(X_train_proc, columns=self.feature_names, index=train_idx).to_csv(self.output_paths["X_TRAIN"], index=False)
        pd.DataFrame(X_test_proc, columns=self.feature_names, index=test_idx).to_csv(self.output_paths["X_TEST"], index=False)
        y_train.to_csv(self.output_paths["Y_TRAIN"], index=False)
        y_test.to_csv(self.output_paths["Y_TEST"], index=False)
        logger.success("Preprocesamiento completado y archivos guardados.")

    def run(self):
        self.load_data()
        self.separate_variables()
        self.encode_features()
        X_train_proc, X_test_proc, y_train, y_test, train_idx, test_idx = self.split_and_transform()
        self.save_outputs(X_train_proc, X_test_proc, y_train, y_test, train_idx, test_idx)