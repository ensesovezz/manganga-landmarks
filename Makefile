# manganga-landmarks — atajos de desarrollo
#
# Entorno local (macOS Intel): Python 3.11 vía uv, PyTorch CPU.

PYTHON ?= .venv/bin/python
UV ?= uv

.PHONY: help setup test validate smoke train detect clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Crea el entorno virtual e instala dependencias
	$(UV) python install 3.11
	$(UV) venv --python 3.11 .venv
	$(UV) pip install --python $(PYTHON) -r requirements.txt
	$(UV) pip install --python $(PYTHON) -e . --no-deps

test: ## Corre los tests
	$(PYTHON) -m pytest tests/ -q

validate: ## Valida el ground truth de masks/ y guarda overlays de las peores
	$(PYTHON) -m manganga.cli validate-data --mask-dir masks --dump-worst 12 -o out

smoke: ## Entrenamiento de prueba en CPU (rápido, calidad baja)
	$(PYTHON) -m manganga.cli train --data-dir . --size 256 --base 16 \
		--max-train-samples 300 --epochs 2 --batch-size 4 --metric-every 1 \
		--metric-samples 20 -o out/checkpoints-smoke

train: ## Entrenamiento completo (usar GPU / Colab)
	$(PYTHON) -m manganga.cli train --data-dir . --epochs 40 --batch-size 16 --device cuda

detect: ## Ejemplo: detectar en una imagen (IMG=ala.jpg)
	$(PYTHON) -m manganga.cli detect -i $(IMG) -o out/

clean: ## Borra artefactos locales
	rm -rf out/ .pytest_cache/ **/__pycache__/
