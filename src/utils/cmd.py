import subprocess
import sys
from loguru import logger

def run_cmd(cmd: list):
    """Ejecuta un comando de shell y muestra logs."""
    try:
        logger.info(f"Ejecutando: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error ejecutando comando: {e}")
        sys.exit(1)

def run_cmd_output(cmd: list) -> str:
    """Ejecuta un comando de shell y devuelve su salida como texto."""
    try:
        logger.info(f"Ejecutando (capturando salida): {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Error ejecutando comando: {e}")
        sys.exit(1)