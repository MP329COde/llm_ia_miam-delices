# Makefile — Raccourcis pour tester, lancer et expérimenter avec llm_ia_miam-delices.
#
# Usage : make <cible>
# Exemple : make test      → lance la suite de tests
#           make run        → démarre l'API FastAPI en mode développement
#           make help       → affiche ce message

PYTHON  ?= python
PYTEST  ?= python -m pytest
UVICORN ?= uvicorn
VENV    = .venv
REQ     = requirements.txt

# Détection de l'environnement virtuel (si présent)
PYTHON_EXEC := $(shell [ -f "$(VENV)/bin/python" ] && echo "$(VENV)/bin/python" || echo "$(PYTHON)")
PYTEST_EXEC := $(PYTHON_EXEC) -m pytest
UVICORN_EXEC:= $(PYTHON_EXEC) -m uvicorn

.PHONY: help install test test-cov run detect train data-sample lint clean

## ─── Aide ──────────────────────────────────────────────────────────────────

help:  ## Affiche les cibles disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

## ─── Installation ──────────────────────────────────────────────────────────

install:  ## Installe les dépendances (via pip dans l'env courant)
	pip install -r $(REQ)
	pip install pytest httpx python-multipart

## ─── Tests ─────────────────────────────────────────────────────────────────

test:  ## Lance tous les tests unitaires et d'intégration
	$(PYTEST_EXEC) tests/ -v

test-cov:  ## Lance les tests avec rapport de couverture (nécessite pytest-cov)
	$(PYTEST_EXEC) tests/ -v --cov=vision --cov=data --cov=api \
		--cov-report=term-missing --cov-report=html:htmlcov

test-detect:  ## Tests uniquement pour vision/detect.py
	$(PYTEST_EXEC) tests/test_detect.py -v

test-data:  ## Tests uniquement pour data/convert_to_yolo.py
	$(PYTEST_EXEC) tests/test_convert_to_yolo.py -v

test-api:  ## Tests uniquement pour api/server.py
	$(PYTEST_EXEC) tests/test_api.py -v

## ─── Développement ─────────────────────────────────────────────────────────

run:  ## Démarre l'API FastAPI en mode rechargement automatique (port 8000)
	$(UVICORN_EXEC) api.server:app --reload --port 8000

## ─── Expérimentation ───────────────────────────────────────────────────────

detect:  ## Teste la détection sur une image (IMAGE=chemin/image.jpg)
ifndef IMAGE
	@echo "Usage : make detect IMAGE=chemin/vers/image.jpg [CONF=0.25] [MODEL=yolov8n.pt]"
	@echo "Exemple : make detect IMAGE=data/sample/sample.jpg"
else
	$(PYTHON_EXEC) vision/detect.py \
		--image $(IMAGE) \
		--conf  $(or $(CONF),0.25) \
		--model $(or $(MODEL),yolov8n.pt)
endif

detect-save:  ## Détection avec export JSON+CSV (IMAGE=… JSON=… CSV=…)
ifndef IMAGE
	@echo "Usage : make detect-save IMAGE=img.jpg JSON=out.json CSV=out.csv"
else
	$(PYTHON_EXEC) vision/detect.py \
		--image $(IMAGE) \
		--conf  $(or $(CONF),0.25) \
		--model $(or $(MODEL),yolov8n.pt) \
		--save-json $(or $(JSON),results.json) \
		--save-csv  $(or $(CSV),results.csv)
	@echo "→ JSON : $(or $(JSON),results.json)"
	@echo "→ CSV  : $(or $(CSV),results.csv)"
endif

## ─── Données ────────────────────────────────────────────────────────────────

data-coco:  ## Convertit des annotations COCO → YOLO (COCO=annot.json OUT=yolo_out/)
ifndef COCO
	@echo "Usage : make data-coco COCO=annotations.json OUT=yolo_out"
else
	$(PYTHON_EXEC) data/convert_to_yolo.py \
		--coco $(COCO) \
		--output-dir $(or $(OUT),yolo_out)
endif

data-csv:  ## Convertit un CSV d'annotations → YOLO (CSV=… IMGS=… OUT=…)
ifndef CSV
	@echo "Usage : make data-csv CSV=boxes.csv IMGS=images/ OUT=yolo_out"
else
	$(PYTHON_EXEC) data/convert_to_yolo.py \
		--csv         $(CSV) \
		--images-dir  $(or $(IMGS),images) \
		--output-dir  $(or $(OUT),yolo_out)
endif

data-sample:  ## Génère des données d'exemple dans data/sample/ pour expérimenter
	$(PYTHON_EXEC) data/sample/generate_sample.py

## ─── Entraînement ───────────────────────────────────────────────────────────

train:  ## Lance l'entraînement YOLOv8 (voir train/yolo_train.sh pour les variables)
	bash train/yolo_train.sh

## ─── Nettoyage ──────────────────────────────────────────────────────────────

clean:  ## Supprime les fichiers temporaires et artefacts de build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage 2>/dev/null || true
	@echo "Nettoyage terminé."
