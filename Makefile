#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = MLops_E38_F2
PYTHON_VERSION = 3.13.7
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt --quiet

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run tests
.PHONY: test
test:
	python -m pytest tests
## Download Data from storage system
.PHONY: sync_data_down
sync_data_down:
	gsutil -m rsync -r gs://DVC/data/ data/
	

## Upload Data to storage system
.PHONY: sync_data_up
sync_data_up:
	gsutil -m rsync -r data/ gs://DVC/data/

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

#################################################################################
# MLflow Server                                                                 #
#################################################################################

## Levanta MLflow Server local en http://localhost:5000
.PHONY: mlflow-server
mlflow-server:
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns \
		--host 0.0.0.0 \
		--port 5000
	
#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m src.data.dataset

## Make clean
.PHONY: clean_data
clean_data:
	$(PYTHON_INTERPRETER) -m src.data.cleaning

## Make FE
.PHONY: FE
FE:
	$(PYTHON_INTERPRETER) -m src.data.features

## Make train
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m src.modeling.train

## Make prepare: ejecuta data → clean_data → FE
.PHONY: prepare
prepare: data clean_data FE

## Make predict
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) -m src.modeling.predict

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

#################################################################################
# DVC COMMANDS                                                                  #
#################################################################################

## Reproduce todo el pipeline según dvc.yaml
.PHONY: dvc_repro
dvc_repro:
	dvc repro

## Sube los datos versionados al remoto (GDrive/S3)
.PHONY: dvc_push
dvc_push:
	dvc push

## Descarga los datos versionados del remoto
.PHONY: dvc_pull
dvc_pull:
	dvc pull

## Verifica qué etapas están desactualizadas
.PHONY: dvc_status
dvc_status:
	dvc status