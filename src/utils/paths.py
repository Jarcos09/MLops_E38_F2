from pathlib import Path

def ensure_path(path_str: str | Path) -> Path:
    """
    Convierte un string o Path en objeto Path y crea los directorios necesarios.
    
    - Si la ruta es un directorio, lo crea directamente.
    - Si la ruta es un archivo, crea su carpeta contenedora.
    
    Args:
        path_str (str | Path): Ruta (archivo o carpeta) a convertir/crear.
    
    Returns:
        Path: Objeto Path garantizado (no crea el archivo, solo el directorio padre).
    """
    path = Path(path_str)

    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)

    return path