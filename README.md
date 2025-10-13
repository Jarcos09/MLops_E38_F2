# E38_Fase_2

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Fase 2 Avance de Proyecto, Gestion del Proyecto de Machine Learning

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         MLFlow/DVC and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── MLFlow/DVC   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes MLFlow/DVC a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
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

## Roles del Equipo
| Integrante | Matrícula | Rol |
|---|---|---|
| Jaime Alejandro Mendívil Altamirano| `A01253316` | SRE / DevOps |
| Christian Erick Mercado Flores | `A00841954` | Software Engineer  |
| Saul Mora Perea | `A01796295` | Data Engineer  |
| Juan Carlos Pérez Nava | `A01795941` | Data Scientist  |
| Mario Javier Soriano Aguilera | `A01384282` | ML Engineer  |

--------

## Instalar paqueterías
```bash
pip install -r requirements.txt --quiet
```
## Clonar repositorio
```bash
git clone https://github.com/Jarcos09/MLops_E38_F2.git
cd MLops_E38/
```
