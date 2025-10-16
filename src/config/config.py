from omegaconf import OmegaConf
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from dataclasses import dataclass
import sys

load_dotenv()

# Configuración global del logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>",
    colorize=True
)


# Carga de Configuración
conf = OmegaConf.load("params.yaml")
conf = OmegaConf.create(OmegaConf.to_yaml(conf, resolve=True))

# Determinar la Raíz del Proyecto
try:
    PROJ_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    logger.warning("No se pudo determinar PROJ_ROOT usando __file__. Asumiendo directorio actual.")
    PROJ_ROOT = Path.cwd()

sys.path.append(str(PROJ_ROOT))

# Rutas base del proyecto
@dataclass(frozen=True)
class ProjectPaths:
    """Contiene las rutas base de directorios del proyecto."""

    DATA: Path = PROJ_ROOT / conf.paths.data
    RAW: Path = PROJ_ROOT / conf.paths.raw
    INTERIM: Path = PROJ_ROOT / conf.paths.interim
    PROCESSED: Path = PROJ_ROOT / conf.paths.processed
    EXTERNAL: Path = DATA / conf.paths.external
    MODELS: Path = PROJ_ROOT / conf.paths.models
    REPORTS: Path = PROJ_ROOT / conf.paths.reports
    FIGURES: Path = REPORTS / conf.paths.figures

PROJECT_PATHS = ProjectPaths()

# Rutas de Limpieza (CLEANING)
@dataclass(frozen=True)
class CleaningPaths:
    """Contiene las rutas de archivos de Cleaning."""
    
    INPUT_FILE: Path = PROJECT_PATHS.RAW / conf.cleaning.input_file
    OUTPUT_FILE: Path = PROJECT_PATHS.INTERIM / conf.cleaning.output_file


CLEANING_PATHS = CleaningPaths()

# Rutas de Descarga (DOWNLOAD)
@dataclass(frozen=True)
class DownloadPaths:
    """Contiene las rutas de archivos de Download."""
    
    DATASET_FILE: Path = PROJECT_PATHS.RAW / conf.download.dataset_filename

DOWNLOAD_PATHS = DownloadPaths()

# Rutas de Preprocesamiento
@dataclass(frozen=True)
class PreprocessingPaths:
    """Contiene todas las rutas de archivos de preprocesamiento."""
    
    # Entrada es la salida de limpieza
    INPUT_FILE: Path = PROJECT_PATHS.INTERIM / conf.preprocessing.input_file
    
    # Rutas de Salida de Sets (X_train, y_test, etc.)
    X_TRAIN: Path = PROJECT_PATHS.PROCESSED / conf.preprocessing.output_files.x_train
    X_TEST: Path = PROJECT_PATHS.PROCESSED / conf.preprocessing.output_files.x_test
    Y_TRAIN: Path = PROJECT_PATHS.PROCESSED / conf.preprocessing.output_files.y_train
    Y_TEST: Path = PROJECT_PATHS.PROCESSED / conf.preprocessing.output_files.y_test

PREPROCESSING_PATHS = PreprocessingPaths()

# Rutas de Entrenamiento (TRAINING)
@dataclass(frozen=True)
class TrainingPaths:
    """Contiene las rutas de archivos de Training."""
    
    MODEL_FILE: Path = PROJECT_PATHS.MODELS / conf.training.model_filename

TRAINING_PATHS = TrainingPaths()