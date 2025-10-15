from omegaconf import OmegaConf
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

conf = OmegaConf.load("params.yaml")
conf = OmegaConf.create(OmegaConf.to_yaml(conf, resolve=True))

# Constantes
PROJ_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJ_ROOT / conf.paths.data
RAW_DATA_DIR = PROJ_ROOT / conf.paths.raw
INTERIM_DATA_DIR = PROJ_ROOT/ conf.paths.interim
PROCESSED_DATA_DIR = PROJ_ROOT / conf.paths.processed
EXTERNAL_DATA_DIR = DATA_DIR / conf.paths.external
MODELS_DIR = PROJ_ROOT / conf.paths.models
REPORTS_DIR = PROJ_ROOT / conf.paths.reports
FIGURES_DIR = REPORTS_DIR / conf.paths.figures

CLEAN_INPUT_PATH = RAW_DATA_DIR / conf.cleaning.input_file
CLEAN_OUTPUT_PATH = INTERIM_DATA_DIR / conf.cleaning.output_file

DOWNLOAD_DATASET_FILE = RAW_DATA_DIR / conf.download.dataset_filename

PREPROCESSING_INPUT_FILE = INTERIM_DATA_DIR / conf.preprocessing.input_file
PREPROCESSING_OUTPUT_XTRAIN = PROCESSED_DATA_DIR / conf.preprocessing.output_files.x_train
PREPROCESSING_OUTPUT_XTEST = PROCESSED_DATA_DIR / conf.preprocessing.output_files.x_test
PREPROCESSING_OUTPUT_YTRAIN = PROCESSED_DATA_DIR / conf.preprocessing.output_files.y_train
PREPROCESSING_OUTPUT_YTEST = PROCESSED_DATA_DIR / conf.preprocessing.output_files.y_test