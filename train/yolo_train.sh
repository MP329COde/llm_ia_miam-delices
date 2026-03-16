#!/usr/bin/env bash
# Script: train/yolo_train.sh — Entraînement YOLOv8 pour la détection d'ingrédients.
set -euo pipefail

# Paramètres par défaut (surchargables via variables d'environnement)
DATA_CONFIG=${DATA_CONFIG:-train/yolo_train.yaml}
MODEL_WEIGHTS=${MODEL_WEIGHTS:-yolov8n.pt}
EPOCHS=${EPOCHS:-100}
IMGSZ=${IMGSZ:-640}
BATCH=${BATCH:-0} # 0 = auto batch size
PROJECT=${PROJECT:-runs/detect}
NAME=${NAME:-exp}

echo "Configuration :"
echo "- data:   $DATA_CONFIG"
echo "- model:  $MODEL_WEIGHTS"
echo "- epochs: $EPOCHS"
echo "- imgsz:  $IMGSZ"
echo "- batch:  $BATCH"
echo "- project/name: $PROJECT/$NAME"

if ! command -v yolo >/dev/null 2>&1; then
  echo "La commande 'yolo' n'est pas disponible. Activez l'environnement (.venv) et installez ultralytics."
  exit 1
fi

yolo detect train \
  data="$DATA_CONFIG" \
  model="$MODEL_WEIGHTS" \
  epochs="$EPOCHS" \
  imgsz="$IMGSZ" \
  batch="$BATCH" \
  project="$PROJECT" \
  name="$NAME" \
  save_json=True \
  exist_ok=True

echo "Entraînement terminé. Les checkpoints et métriques sont dans $PROJECT/$NAME"
