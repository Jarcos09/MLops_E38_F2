from omegaconf import OmegaConf
from dotenv import load_dotenv
from loguru import logger
import sys
from src.utils import cmd

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

# Configuración del DVC
def dvc_setup():
    """Inicializa y configura DVC automáticamente usando conf.dvc"""
    remote_url = conf.dvc.remote_url
    client_id = conf.dvc.client_id
    client_secret = conf.dvc.client_secret

    if not all([remote_url, client_id, client_secret]):
        logger.error("Faltan variables en params.yaml o .env para DVC.")
        logger.info(f"remote_url={remote_url}, client_id={client_id}, client_secret={bool(client_secret)}")
        sys.exit(1)

    logger.info("Inicializando configuración de DVC...")

    # Inicializa DVC
    cmd.run_cmd(["dvc", "init", "-f"])

    # Verifica si ya existe el remoto
    existing_remotes = cmd.run_cmd_output(["dvc", "remote", "list"])

    if "data" not in existing_remotes:
        cmd.run_cmd(["dvc", "remote", "add", "-d", "data", remote_url])
    else:
        logger.info("Remoto 'data' ya configurado, omitiendo creación.")

    # Configura credenciales del remoto
    cmd.run_cmd(["dvc", "remote", "modify", "data", "gdrive_client_id", client_id])
    cmd.run_cmd(["dvc", "remote", "modify", "data", "gdrive_client_secret", client_secret])

    # Verifica configuración final
    cmd.run_cmd(["dvc", "remote", "list"])
    logger.success("Configuración de DVC completada con éxito.")

if __name__ == "__main__":
    dvc_setup()