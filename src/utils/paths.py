from pathlib import Path

# Función para asegurar la existencia de directorios a partir de una ruta

# Convierte la ruta a objeto Path y crea directorios según corresponda
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

    # 2) Si es archivo, crear carpeta contenedora
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # 3) Si es directorio, crear directamente
        path.mkdir(parents=True, exist_ok=True)

    return path