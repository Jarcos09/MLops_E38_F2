# E38_Fase_2

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Fase 2 Avance de Proyecto, Gestion del Proyecto de Machine Learning

## Project Organization

```
├── LICENSE                         <- Open-source license if one is chosen
├── Makefile                        <- Makefile with convenience commands like `make data` or `make train`
├── README.md                       <- The top-level README for developers using this project.
├── params.yaml                     <- Centralized configuration file for pipeline parameters.
├── data                
│   ├── external                    <- Data from third party sources.
│   ├── interim                     <- Intermediate data that has been transformed.
│   ├── processed                   <- The final, canonical data sets for modeling.
│   └── raw                         <- The original, immutable data dump.
│               
├── docs                            <- A default mkdocs project; see www.mkdocs.org for details
│               
├── models                          <- Trained and serialized models, model predictions, or model summaries
│               
├── notebooks                       <- Jupyter notebooks. Naming convention is a number (for ordering),
│                                      the creator's initials, and a short `-` delimited description, e.g.
│                                      `1.0-jqp-initial-data-exploration`.
│               
├── pyproject.toml                  <- Project configuration file with package metadata for 
│                                      MLFlow/DVC and configuration for tools like black
│               
├── references                      <- Data dictionaries, manuals, and all other explanatory materials.
│               
├── reports                         <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                     <- Generated graphics and figures to be used in reporting
│               
├── requirements.txt                <- The requirements file for reproducing the analysis environment, e.g.
│                                      generated with `pip freeze > requirements.txt`
│               
├── setup.cfg                       <- Configuration file for flake8
│               
└── src                             <- Source code for the project
    ├── __init__.py                 <- Makes `src` a Python module
    ├── utils
    │   ├── __init__.py
    │   ├── cmd.py                  <- Functions to execute cmd commands
    │   └── paths.py                <- Paths manager to create and ensure directories
    ├── config
    │   ├── __init__.py
    │   ├── dvc_setup.py            <- Functions to set dvc repos
    │   └── config.py               <- Store useful variables and configuration
    ├── data
    │   ├── __init__.py
    │   ├── clean_dataset.py        <- Script to clean raw data
    │   ├── cleaning.py             <- Main cleaning scripts
    │   ├── dataset.py              <- Scripts to download or generate data
    │   ├── download_dataset.py     <- Scripts to fetch datasets from external sources
    │   ├── features.py             <- Code to create features for modeling
    │   └── preprocess_data.py      <- Preprocessing pipelines for ML
    └── modeling
        ├── __init__.py
        ├── plots.py                <- Code to create visualizations
        ├── predict.py              <- Code to run model inference with trained models
        ├── train_model.py          <- Model training logic and MLFlow integration
        └── train.py                <- Entry point to train models
```

--------

# Fase 2 | Avance de Proyecto
# Equipo 38

En esta actividad se continuará con el desarrollo del proyecto, dando seguimiento a los avances realizados en la Fase 1. Se mantendrá la propuesta de valor, el análisis elaborado con el ML Canvas, así como los datos, modelos y experimentos previamente desarrollados. El objetivo ahora es estructurar el proyecto de Machine Learning de forma profesional, aplicando buenas prácticas como la refactorización del código, el control de versiones, el seguimiento de experimentos, el registro de métricas y modelos, y el aseguramiento de la reproducibilidad.

--------

## 🎯 Objetivos

- Continuar con el desarrollo de proyectos de Machine Learning, a partir de los requerimientos, una propuesta de valor y un conjunto de datos preprocesados.
- Estructurar proyectos de Machine Learning de manera organizada (utilizando el template de Cookiecutter)
- Aplicar buenas prácticas de codificación en cada etapa del pipeline y realizar Refactorización del código.
- Registrar métricas y aplicar control de versiones  a los experimentos utilizando herramientas de loging y tracking  (MLFlow/DVC)
- Visualizar y comparar resultados (métricas) y gestionar el registro de los modelos (Data Registry MLFlow/DVC)

--------

## 👥 Roles del Equipo
| Integrante | Matrícula | Rol |
|---|---|---|
| Jaime Alejandro Mendívil Altamirano| `A01253316` | SRE / DevOps |
| Christian Erick Mercado Flores | `A00841954` | Software Engineer  |
| Saul Mora Perea | `A01796295` | Data Engineer  |
| Juan Carlos Pérez Nava | `A01795941` | Data Scientist  |
| Mario Javier Soriano Aguilera | `A01384282` | ML Engineer  |

--------

## 📦 Instalar paqueterías
```bash
pip install -r requirements.txt --quiet
```
## 💼 Clonar repositorio
```bash
git clone https://github.com/Jarcos09/MLops_E38_F2.git
cd MLops_E38_F2/
```

--------

## 📚 Makefile

Descargar Dataset:
```bash
make data
```

Realizar limpieza del Dataset:
```bash
make clean_data
```

Realizar FE:
```bash
make FE
```

Ejecuta (data → clean_data → FE):
```bash
make prepare
```

Ejecutar localmente servidor de MLFlow:
```bash
make mlflow-server
```

Realizar entrenamiento:
```bash
make train
```

Realizar preducción:
```bash
make predict
```

Configuración completa de DVC remoto:
```bash
make dvc_setup
```

Ejecutar el pipeline completo de DVC (data → clean → FE → train):
```bash
make dvc_repro
```

Subir los outputs del pipeline al remoto:
```bash
make dvc_push
```

Descargar los datos versionados del remoto:
```bash
make dvc_pull
```

Verificar qué etapas del pipeline están desactualizadas:
```bash
make dvc_status
```

--------

## 🧠 MLflow

**MLflow** es una herramienta para gestionar el ciclo de vida de modelos de Machine Learning: rastrea experimentos, guarda métricas y versiona modelos.

---

### Iniciar servidor local

Se puede utilizar el comando:
```bash
make dvc_setup
```

También se puede ejecutar el servidor en modo local con SQLite y carpeta `mlruns`:
```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000
````

### Interfaz
http://localhost:5000

### Integración en el Proyecto
* `train_model.py`: Registra métricas, parámetros y modelos (Random Forest, XGBoost).

* `predict_model.py`: Usa modelos registrados para generar predicciones.

* `config/config.py`: Define la URI de tracking (mlflow_tracking_uri).

--------

## 💾 DVC

### Inicialización de Repositorio DVC

Se puede utilizar el comando:
```bash
make mlflow-server
```

También, se puede inicializar manualmente de la siguiente manera:
```bash
dvc init
```
### GDRIVE
#### Agregar Repositorio DVC (GDrive)
```bash
dvc remote add -d data "$GDRIVE_REMOTE_URL"
```

#### Configuración de DVC (GDrive)
```bash
dvc remote modify data gdrive_client_id "$GDRIVE_CLIENT_ID"
dvc remote modify data gdrive_client_secret "$GDRIVE_CLIENT_SECRET"
```

### AWS
#### Agregar Repositorio DVC (AWS)
```bash
dvc remote add -d data "$AWS_REMOTE_URL"
```

#### Configuración de DVC (AWS)
```bash
dvc remote modify team_remote region "$AWS_REGION"
dvc remote modify team_remote profile "$AWS_PROFILE"
```

### Verificar Repositorios DVC Configurados
```bash
dvc remote list
```

### Repositorio DVC (GDrive)
[Carpeta Principal del Proyecto en Google Drive](https://drive.google.com/drive/u/2/folders/1VnjNYOpP2uSaaUtFdRzW45iwZJUbt-5v)

--------