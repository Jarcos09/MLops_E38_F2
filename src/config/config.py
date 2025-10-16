from omegaconf import OmegaConf
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from dataclasses import dataclass
import sys

# Carga de variables de entorno
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