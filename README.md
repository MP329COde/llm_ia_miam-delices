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
python data/convert_to_yolo.py --coco chemin/annotations.json --images-dir chemin/images --output-dir sortie_yolo
```
Le script accepte aussi un CSV (colonnes `file,xmin,ymin,xmax,ymax,class_name`).

## Structure de base actuelle
- `install.sh` : création d'environnement virtuel et installation des dépendances.
- `requirements.txt` / `environment.yml` : listes des dépendances Python (CPU/GPU).
- `vision/detect.py` : détection d'ingrédients via YOLOv8 avec fallback CPU.
- `api/server.py` : API FastAPI minimale pour l'analyse d'images.
- `train/yolo_train.sh` : script shell pour entraîner un modèle de détection.
- `data/convert_to_yolo.py` : conversion COCO/CSV vers labels YOLO.

## Prochaines étapes (feuille de route)
- Ajouter la classification timm et la génération de légendes (BLIP).
- Implémenter le pipeline RAG (ChromaDB + embeddings CLIP).
- Intégrer un LLM local (llama.cpp / gpt4all) pour générer des recettes.
- Créer l'interface web (React) pour l'upload et le feedback utilisateur.
