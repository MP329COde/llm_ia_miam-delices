# llm_ia_miam-delices

## Présentation

IA culinaire locale, gratuite et open-source : détection d'ingrédients, génération de légende, suggestion de recettes (RAG) et génération de recettes (LLM). Tout fonctionne en CPU/GPU, sans services propriétaires.

## Architecture

```
┌──────────────┐   image    ┌────────────────────┐
│  API FastAPI │ ────────►  │  vision/detect.py  │  YOLOv8 — détection d'ingrédients
│  (port 8000) │            └────────────────────┘
│              │   image    ┌────────────────────┐
│              │ ────────►  │  vision/caption.py │  BLIP — légende textuelle
│              │            └────────────────────┘
│              │ ingredients┌────────────────────┐
│              │ ────────►  │  rag/pipeline.py   │  ChromaDB/TF-IDF — suggestions de recettes
│              │            └────────────────────┘
│              │ ingredients┌────────────────────┐
│              │ ────────►  │  llm/generate.py   │  llama.cpp/gpt4all/stub — génération
└──────────────┘            └────────────────────┘
```

## Structure du projet

```
llm_ia_miam-delices/
├── api/
│   └── server.py            # API FastAPI (8 endpoints)
├── vision/
│   ├── detect.py            # Détection YOLOv8
│   └── caption.py           # Légende BLIP
├── rag/
│   ├── recipes.py           # 20 recettes embarquées + chargement JSON
│   └── pipeline.py          # Pipeline RAG (ChromaDB ou TF-IDF fallback)
├── llm/
│   └── generate.py          # Génération de recettes (llama.cpp / gpt4all / stub)
├── data/
│   ├── convert_to_yolo.py   # Conversion COCO/CSV → labels YOLO
│   ├── import_images.py     # Import dossier/ZIP/URL → dataset train/val
│   └── sample/              # Données synthétiques d'exemple
├── train/
│   ├── train.py             # Wrapper Python d'entraînement YOLOv8
│   ├── yolo_train.sh        # Script shell alternatif
│   └── yolo_train.yaml      # Configuration dataset YOLO
├── tests/                   # Suite de tests (179 tests)
├── Makefile                 # Toutes les commandes en un seul endroit
├── requirements.txt         # Dépendances Python
└── environment.yml          # Conda (avec support CUDA optionnel)
```

## Prérequis

- Python 3.10+
- (Optionnel) GPU NVIDIA + CUDA 11.8 pour accélération

## Installation

```bash
chmod +x install.sh
./install.sh
source .venv/bin/activate
# ou
make install
```

### Windows
```powershell
powershell -ExecutionPolicy Bypass -File .\install_windows.ps1
.\.venv\Scripts\Activate.ps1
```

## Démarrage de l'API

```bash
make run
# ou
uvicorn api.server:app --reload --port 8000
```

### Endpoints disponibles

| Méthode | Endpoint          | Description |
|---------|-------------------|-------------|
| GET     | `/health`         | Santé du service |
| POST    | `/analyse`        | Détection YOLOv8 sur une image |
| POST    | `/caption`        | Légende BLIP d'une image |
| POST    | `/suggest-recipes`| Suggestions de recettes (RAG) |
| POST    | `/generate-recipe`| Génération de recette (LLM) |
| POST    | `/analyse-full`   | Pipeline complet en un appel |
| POST    | `/upload-dataset` | Import d'images dans le dataset |
| POST    | `/feedback`       | Collecte de feedback utilisateur |

Documentation interactive : http://localhost:8000/docs

### Exemples curl

```bash
# Santé
curl http://localhost:8000/health

# Détection d'ingrédients
curl -X POST http://localhost:8000/analyse \
     -F "file=@data/sample/sample.jpg" \
     -F "conf=0.25"

# Suggestions de recettes
curl -X POST http://localhost:8000/suggest-recipes \
     -H "Content-Type: application/json" \
     -d '{"ingredients": ["tomate", "carotte", "oignon"], "top_k": 5}'

# Génération de recette (stub/démo)
curl -X POST http://localhost:8000/generate-recipe \
     -H "Content-Type: application/json" \
     -d '{"ingredients": ["tomate", "courgette"], "backend": "stub"}'

# Pipeline complet
curl -X POST http://localhost:8000/analyse-full \
     -F "file=@data/sample/sample.jpg" \
     -F "do_suggest=true" \
     -F "do_generate=true"

# Import d'images dans le dataset
curl -X POST http://localhost:8000/upload-dataset \
     -F "files=@photo1.jpg" \
     -F "files=@photo2.jpg" \
     -F "output_dir=data/dataset" \
     -F "val_split=0.2"
```

## Tests — Vérifier ce qui fonctionne

```bash
make test               # 179 tests (sans GPU requis)
make test-cov           # tests + rapport de couverture HTML
make test-api           # API uniquement
make test-rag           # RAG uniquement
make test-llm           # LLM uniquement
make test-caption       # BLIP uniquement
make test-detect        # YOLOv8 détection
make test-import        # Import d'images
make test-train         # Entraînement
```

## Expérimenter avec des données d'exemple

```bash
# 1. Générer l'image synthétique + annotations
make data-sample

# 2. Tester la détection CLI
make detect IMAGE=data/sample/sample.jpg

# 3. Obtenir des suggestions de recettes en CLI
make rag-index INGREDIENTS="tomate, carotte, oignon"

# 4. Générer une recette (stub, sans LLM)
make generate INGREDIENTS="tomate, courgette, ail"
```

## Machine Learning — Ajouter vos données et entraîner

### 1. Importer des images

```bash
# Depuis un dossier local
make import-folder SRC=mes_photos/ OUT=data/dataset VAL=0.2 SIZE=640

# Depuis une archive ZIP
make import-zip ZIP=photos.zip OUT=data/dataset VAL=0.2

# Depuis la CLI Python
python data/import_images.py --source mes_photos/ --output-dir data/dataset --val-split 0.2
python data/import_images.py --zip photos.zip --output-dir data/dataset
python data/import_images.py --urls urls.txt --output-dir data/dataset
```

### 2. Annoter les images

Utilisez l'un de ces outils gratuits :
- **LabelImg** : `pip install labelImg && labelImg`
- **CVAT** : https://cvat.ai (cloud gratuit)
- **Roboflow** : https://roboflow.com (export COCO ou YOLO)

### 3. Convertir les annotations

```bash
# Depuis COCO JSON (LabelImg, CVAT, Roboflow)
make data-coco COCO=annotations.json OUT=data/dataset/labels

# Depuis CSV (file,xmin,ymin,xmax,ymax,class_name)
make data-csv CSV=boxes.csv IMGS=data/dataset/images/train OUT=data/dataset/labels
```

### 4. Configurer le dataset

Éditez `train/yolo_train.yaml` :
```yaml
path: data/dataset
train: images/train
val: images/val
names:
  0: tomate
  1: carotte
  2: courgette
  # ... ajoutez vos classes
```

### 5. Lancer l'entraînement

```bash
make train DATA=train/yolo_train.yaml MODEL=yolov8n.pt EPOCHS=50 BATCH=16

# Ou avec le wrapper Python directement
python train/train.py train \
    --data train/yolo_train.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --batch 16 \
    --device cpu

# Évaluer le modèle entraîné
make train-val MODEL_PT=runs/detect/train/weights/best.pt
```

## LLM local — Générer des recettes avec IA

### Avec llama.cpp (recommandé)

```bash
pip install llama-cpp-python
# Télécharger un modèle GGUF depuis Hugging Face, ex: Mistral 7B Q4
python llm/generate.py \
    --ingredients "tomate, carotte, oignon" \
    --backend llama_cpp \
    --model-path /chemin/vers/mistral-7b-q4.gguf
```

### Avec GPT4All

```bash
pip install gpt4all
python llm/generate.py \
    --ingredients "tomate, carotte" \
    --backend gpt4all \
    --model-path /chemin/vers/model.gguf
```

### Mode stub (démo sans LLM)

```bash
python llm/generate.py --ingredients "tomate, courgette" --backend stub
# ou via l'API
make generate INGREDIENTS="tomate, courgette"
```

## RAG — Chercher des recettes par ingrédients

```bash
# CLI (20 recettes embarquées + TF-IDF fallback)
python rag/pipeline.py --ingredients "tomate, basilic" --top 3

# Avec ChromaDB + sentence-transformers (plus précis)
pip install chromadb sentence-transformers
python rag/pipeline.py --ingredients "brocoli, carotte" --top 5

# Avec vos propres recettes JSON
python rag/pipeline.py --recipes-json mes_recettes.json --ingredients "poulet"
```

### Format recettes JSON personnalisées

```json
[
  {
    "id": "ma_recette_01",
    "title": "Soupe maison",
    "ingredients": ["tomate", "oignon", "ail"],
    "instructions": "Cuire tous les légumes...",
    "tags": ["végétarien", "soupe"]
  }
]
```

## BLIP — Légende d'image

```bash
pip install transformers
python vision/caption.py --image data/sample/sample.jpg
```

> Le modèle `Salesforce/blip-image-captioning-base` (~1 Go) sera téléchargé depuis Hugging Face.

## Performance

| Composant | CPU (recommandé) | GPU (optionnel) |
|-----------|-----------------|-----------------|
| YOLOv8 détection | `yolov8n.pt` | `yolov8s/m/l.pt` |
| BLIP légende | `blip-image-captioning-base` | idem |
| LLM génération | GGUF 4-bit Q4_0 | GGUF 8-bit |
| RAG | TF-IDF (intégré) | ChromaDB + sentence-transformers |

## Prochaines étapes

- [ ] Interface web React (upload + affichage des résultats)
- [ ] Persistance des feedbacks (SQLite)
- [ ] Intégration CLIP pour la recherche d'images similaires
- [ ] Fine-tuning BLIP sur des recettes françaises
- [ ] API de gestion du dataset (labels, classes, statistiques)
