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

.PHONY: help install test test-cov test-detect test-data test-api test-rag test-llm test-caption \
        run detect detect-save data-coco data-csv data-sample import-folder import-zip \
        caption rag-index generate train train-val clean

## ─── Aide ──────────────────────────────────────────────────────────────────

help:  ## Affiche les cibles disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

## ─── Installation ──────────────────────────────────────────────────────────

install:  ## Installe toutes les dépendances
	pip install -r $(REQ)
	pip install scikit-learn

## ─── Tests ─────────────────────────────────────────────────────────────────

test:  ## Lance tous les tests (179 tests)
	$(PYTEST_EXEC) tests/ -v

test-cov:  ## Lance les tests avec rapport de couverture HTML
	$(PYTEST_EXEC) tests/ -v \
		--cov=vision --cov=data --cov=api --cov=rag --cov=llm --cov=train \
		--cov-report=term-missing --cov-report=html:htmlcov

test-detect:  ## Tests vision/detect.py
	$(PYTEST_EXEC) tests/test_detect.py -v

test-data:  ## Tests data/convert_to_yolo.py
	$(PYTEST_EXEC) tests/test_convert_to_yolo.py -v

test-api:  ## Tests api/server.py (anciens + nouveaux endpoints)
	$(PYTEST_EXEC) tests/test_api.py tests/test_api_extended.py -v

test-rag:  ## Tests rag/ (recipes + pipeline)
	$(PYTEST_EXEC) tests/test_rag.py -v

test-llm:  ## Tests llm/generate.py
	$(PYTEST_EXEC) tests/test_llm.py -v

test-caption:  ## Tests vision/caption.py
	$(PYTEST_EXEC) tests/test_caption.py -v

test-import:  ## Tests data/import_images.py
	$(PYTEST_EXEC) tests/test_import_images.py -v

test-train:  ## Tests train/train.py
	$(PYTEST_EXEC) tests/test_train.py -v

## ─── Développement ─────────────────────────────────────────────────────────

run:  ## Démarre l'API FastAPI en mode rechargement automatique (port 8000)
	$(UVICORN_EXEC) api.server:app --reload --port 8000

## ─── Détection (vision) ─────────────────────────────────────────────────────

detect:  ## Détection YOLOv8 sur une image (IMAGE=chemin/image.jpg)
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

caption:  ## Génère une légende BLIP pour une image (IMAGE=chemin/image.jpg)
ifndef IMAGE
	@echo "Usage : make caption IMAGE=chemin/vers/image.jpg"
else
	$(PYTHON_EXEC) vision/caption.py --image $(IMAGE)
endif

## ─── RAG & LLM ──────────────────────────────────────────────────────────────

rag-index:  ## Indexe les recettes embarquées dans ChromaDB/TF-IDF (INGREDIENTS="tomate, carotte")
ifndef INGREDIENTS
	@echo "Usage : make rag-index INGREDIENTS='tomate, carotte, oignon' [TOP=5]"
else
	$(PYTHON_EXEC) rag/pipeline.py \
		--ingredients "$(INGREDIENTS)" \
		--top $(or $(TOP),5)
endif

generate:  ## Génère une recette avec le LLM (INGREDIENTS="tomate, carotte")
ifndef INGREDIENTS
	@echo "Usage : make generate INGREDIENTS='tomate, carotte' [MODEL_PATH=/chemin/model.gguf]"
else
	$(PYTHON_EXEC) llm/generate.py \
		--ingredients "$(INGREDIENTS)" \
		--backend $(or $(BACKEND),auto) \
		$(if $(MODEL_PATH),--model-path $(MODEL_PATH),)
endif

## ─── Données ────────────────────────────────────────────────────────────────

data-coco:  ## Convertit annotations COCO → YOLO (COCO=annot.json OUT=yolo_out/)
ifndef COCO
	@echo "Usage : make data-coco COCO=annotations.json OUT=yolo_out"
else
	$(PYTHON_EXEC) data/convert_to_yolo.py \
		--coco $(COCO) \
		--output-dir $(or $(OUT),yolo_out)
endif

data-csv:  ## Convertit CSV d'annotations → YOLO (CSV=… IMGS=… OUT=…)
ifndef CSV
	@echo "Usage : make data-csv CSV=boxes.csv IMGS=images/ OUT=yolo_out"
else
	$(PYTHON_EXEC) data/convert_to_yolo.py \
		--csv         $(CSV) \
		--images-dir  $(or $(IMGS),images) \
		--output-dir  $(or $(OUT),yolo_out)
endif

data-sample:  ## Génère des données d'exemple dans data/sample/
	$(PYTHON_EXEC) data/sample/generate_sample.py

import-folder:  ## Importe images depuis un dossier → train/val (SRC=dossier OUT=data/dataset)
ifndef SRC
	@echo "Usage : make import-folder SRC=mon_dossier/ OUT=data/dataset [VAL=0.2] [SIZE=640]"
else
	$(PYTHON_EXEC) data/import_images.py \
		--source     $(SRC) \
		--output-dir $(or $(OUT),data/dataset) \
		--val-split  $(or $(VAL),0.2) \
		$(if $(SIZE),--max-size $(SIZE),)
endif

import-zip:  ## Importe images depuis une archive ZIP → train/val (ZIP=archive.zip OUT=data/dataset)
ifndef ZIP
	@echo "Usage : make import-zip ZIP=archive.zip OUT=data/dataset [VAL=0.2]"
else
	$(PYTHON_EXEC) data/import_images.py \
		--zip        $(ZIP) \
		--output-dir $(or $(OUT),data/dataset) \
		--val-split  $(or $(VAL),0.2)
endif

## ─── Entraînement ───────────────────────────────────────────────────────────

train:  ## Lance l'entraînement YOLOv8 (wrapper Python)
	$(PYTHON_EXEC) train/train.py train \
		--data    $(or $(DATA),train/yolo_train.yaml) \
		--model   $(or $(MODEL),yolov8n.pt) \
		--epochs  $(or $(EPOCHS),50) \
		--imgsz   $(or $(IMGSZ),640) \
		--batch   $(or $(BATCH),16)

train-val:  ## Évalue un modèle entraîné (MODEL_PT=runs/detect/train/weights/best.pt)
ifndef MODEL_PT
	@echo "Usage : make train-val MODEL_PT=runs/detect/train/weights/best.pt"
else
	$(PYTHON_EXEC) train/train.py val \
		--model $(MODEL_PT) \
		--data  $(or $(DATA),train/yolo_train.yaml)
endif

## ─── Nettoyage ──────────────────────────────────────────────────────────────

clean:  ## Supprime les artefacts de build et de test
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage rag/db 2>/dev/null || true
	@echo "Nettoyage terminé."
