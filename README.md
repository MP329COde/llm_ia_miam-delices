# llm_ia_miam-delices

## Présentation
Ce projet fournit une base locale, gratuite et open-source pour expérimenter une IA culinaire complète : détection d'ingrédients, génération de légende, RAG sur recettes et API locale. L'objectif est de proposer une chaîne reproductible fonctionnant aussi bien en CPU qu'avec un GPU CUDA si disponible, sans dépendre de services propriétaires.

## Prérequis
- Python 3.10 ou supérieur
- (Optionnel) GPU NVIDIA avec pilotes et `nvidia-smi` pour accélération CUDA
- Git, bash et accès réseau pour télécharger les dépendances open-source

## Installation rapide
```bash
chmod +x install.sh
./install.sh
```
Le script crée un environnement virtuel `.venv`, détecte automatiquement la présence de CUDA et installe les dépendances listées dans `requirements.txt`.

### Installation Windows (PowerShell)
```powershell
powershell -ExecutionPolicy Bypass -File .\install_windows.ps1
.\.venv\Scripts\Activate.ps1
```
Le script Windows est équivalent : il configure la venv, teste `nvidia-smi` et installe les dépendances. En absence de GPU, tout fonctionne en CPU-only (plus lent).

## Démarrage de l'API
```bash
source .venv/bin/activate
uvicorn api.server:app --reload --port 8000
```
L'API FastAPI expose notamment l'endpoint `POST /analyse` pour analyser une image et renvoyer un JSON avec les ingrédients détectés. En l'absence de GPU ou de modèle pré-téléchargé, le flux retombe automatiquement sur un mode CPU.

## Détection rapide en ligne de commande
```bash
source .venv/bin/activate
python vision/detect.py --image chemin/vers/image.jpg --conf 0.25 --save-csv sorties.csv
```

## Entraînement YOLOv8 (détection)
```bash
source .venv/bin/activate
bash train/yolo_train.sh
```
Par défaut, le script utilise `train/yolo_train.yaml` (à adapter à votre dataset) et le modèle `yolov8n.pt` en transfert de learning. Les journaux et checkpoints sont déposés dans `runs/detect/`.

## Conversion d'annotations au format YOLO
```bash
source .venv/bin/activate
python data/convert_to_yolo.py --coco chemin/annotations.json --output-dir sortie_yolo
# Pour un CSV (requiert les images pour calculer les dimensions) :
# python data/convert_to_yolo.py --csv boxes.csv --images-dir chemin/images --output-dir sortie_yolo
```
Le script accepte aussi un CSV (colonnes `file,xmin,ymin,xmax,ymax,class_name`).

## Performance & meilleurs modèles open-source (gratuit)
- Détection rapide : `yolov8n.pt` (CPU/GPU), passer à `yolov8s.pt` ou `yolov8m.pt` si plus de VRAM.
- Légende d'image : `Salesforce/blip-image-captioning-base` (gratuit via transformers).
- Classification : backbones `convnext_base` ou `efficientnet_b3` (timm) avec `pretrained=True`.
- LLM local : modèles GGUF (llama.cpp) ou gpt4all (ex : `gpt4all-falcon-q4_0.gguf`) — gratuits, téléchargeables depuis Hugging Face.
- Optimisations :
  - GPU : augmentez `batch` dans `train/yolo_train.sh` (0 = auto YOLOv8), utilisez `imgsz` réduit (512/448) en cas de VRAM limitée.
  - CPU-only : préférez `yolov8n.pt`, désactivez l'affichage/sauvegarde (`show=False`, `save=False` avec Ultralytics). Un seuil `--conf` plus élevé filtre les boîtes faibles et réduit un peu la post-procession.
  - Activations rapides : PyTorch active `cudnn.benchmark` automatiquement pour lots fixes ; sinon fixez `CUDNN_BENCHMARK=1`.
  - Quantization LLM : Utiliser gguf 4/8 bits pour CPU, GPU si disponible dans llama.cpp/gpt4all.

## Structure de base actuelle
- `install.sh` : création d'environnement virtuel et installation des dépendances.
- `install_windows.ps1` : installation équivalente pour Windows (PowerShell).
- `requirements.txt` / `environment.yml` : listes des dépendances Python (CPU/GPU).
- `vision/detect.py` : détection d'ingrédients via YOLOv8 avec fallback CPU.
- `api/server.py` : API FastAPI minimale pour l'analyse d'images.
- `train/yolo_train.sh` : script shell pour entraîner un modèle de détection.
- `data/convert_to_yolo.py` : conversion COCO/CSV vers labels YOLO.
- `data/sample/` : données d'exemple synthétiques et script de génération.
- `tests/` : suite de tests unitaires et d'intégration (pytest).
- `Makefile` : raccourcis pour tester, lancer, convertir et expérimenter.

## Tests — Vérifier ce qui fonctionne

La suite de tests couvre `vision/detect.py`, `data/convert_to_yolo.py` et `api/server.py`
sans nécessiter de modèle ML ni de GPU.

```bash
# Lancer tous les tests
make test
# ou directement :
python -m pytest tests/ -v

# Cibler un module
make test-detect   # vision/detect.py
make test-data     # data/convert_to_yolo.py
make test-api      # api/server.py

# Avec rapport de couverture (nécessite pytest-cov)
pip install pytest-cov
make test-cov
```

## Expérimenter avec des données d'exemple

```bash
# 1. Générer des images et annotations synthétiques
make data-sample
# Crée data/sample/sample.jpg, annotations_coco.json, annotations.csv

# 2. Convertir les annotations COCO en format YOLO
make data-coco COCO=data/sample/annotations_coco.json OUT=data/sample/yolo_out

# 3. Convertir un CSV d'annotations en format YOLO
make data-csv CSV=data/sample/annotations.csv IMGS=data/sample OUT=data/sample/yolo_out

# 4. Tester la détection sur l'image synthétique (nécessite ultralytics)
make detect IMAGE=data/sample/sample.jpg
```

### Ajouter vos propres données

**Format COCO** : exportez depuis LabelImg, CVAT ou Roboflow au format COCO JSON, puis :
```bash
python data/convert_to_yolo.py --coco vos_annotations.json --output-dir yolo_out/
```

**Format CSV** : créez un fichier avec les colonnes `file,xmin,ymin,xmax,ymax,class_name`
(une ligne par boîte englobante), puis :
```bash
python data/convert_to_yolo.py --csv vos_boites.csv --images-dir dossier_images/ --output-dir yolo_out/
```

Copiez ensuite le dossier `yolo_out/` vers `data/dataset/labels/` et mettez à jour
`train/yolo_train.yaml` avec vos classes avant de lancer `make train`.

## Prochaines étapes (feuille de route)
- Ajouter la classification timm et la génération de légendes (BLIP).
- Implémenter le pipeline RAG (ChromaDB + embeddings CLIP).
- Intégrer un LLM local (llama.cpp / gpt4all) pour générer des recettes.
- Créer l'interface web (React) pour l'upload et le feedback utilisateur.
