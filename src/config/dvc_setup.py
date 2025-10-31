from loguru import logger
import sys
from src.utils import cmd
from src.config.config import conf

# Configuración del DVC
def dvc_gdrive_setup():
    """Inicializa y configura DVC automáticamente usando conf.dvc"""
    remote_url = conf.dvc.gdrive_remote_url
    client_id = conf.dvc.gdrive_client_id
    client_secret = conf.dvc.gdrive_client_secret

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

def dvc_aws_setup():
    """Inicializa y configura DVC automáticamente usando conf.dvc"""
    remote_url = conf.dvc.aws_remote_url
    aws_region = conf.dvc.aws_region
    aws_profile = conf.dvc.aws_profile

    if not all([remote_url, aws_region, aws_profile]):
        logger.error("Faltan variables en params.yaml o .env para DVC.")
        logger.info(f"remote_url={remote_url}, aws_region={aws_region}, aws_profile={bool(aws_profile)}")
        sys.exit(1)

    logger.info("Inicializando configuración de DVC...")

    # Inicializa DVC
    cmd.run_cmd(["dvc", "init", "-f"])

    # Verifica si ya existe el remoto
    existing_remotes = cmd.run_cmd_output(["dvc", "remote", "list"])

    if "data" not in existing_remotes:
        cmd.run_cmd(["dvc", "remote", "add", "-d", "team_remote", remote_url])
    else:
        logger.info("Remoto 'team_remote' ya configurado, omitiendo creación.")

    # Configura credenciales del remoto
    cmd.run_cmd(["dvc", "remote", "modify", "team_remote", "region", aws_region])
    cmd.run_cmd(["dvc", "remote", "modify", "team_remote", "profile", aws_profile])

    # Verifica configuración final
    cmd.run_cmd(["dvc", "remote", "list"])
    logger.success("Configuración de DVC completada con éxito.")

if __name__ == "__main__":
    option = sys.argv[1].lower()
    if option == "gdrive":
        dvc_gdrive_setup()
    elif option == "aws":
        dvc_aws_setup()
    else:
        logger.error(f"Opción desconocida: {option}. Usa 'gdrive' o 'aws'.")