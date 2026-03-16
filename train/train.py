#!/usr/bin/env python
# Fichier: train/train.py — Wrapper Python reproductible pour l'entraînement YOLOv8.
"""
Ce script fournit une interface Python de haut niveau pour entraîner et fine-tuner
un modèle YOLOv8 sur un dataset culinaire.

Il permet de :
- Valider la configuration avant le lancement
- Lancer l'entraînement avec logging structuré
- Sauvegarder les hyperparamètres utilisés dans un JSON reproductible
- Résumer les métriques d'entraînement à la fin

Usage CLI
---------
>>> python train/train.py --data train/yolo_train.yaml --epochs 50
>>> python train/train.py --data train/yolo_train.yaml --model yolov8s.pt --epochs 100 --batch 16

Résultats
---------
Les checkpoints, métriques et logs sont déposés dans `runs/detect/<name>/`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except Exception:
    YOLO = None  # type: ignore[assignment,misc]
    _YOLO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Validation de la configuration
# ---------------------------------------------------------------------------

def validate_config(data_yaml: Path, model_weights: str) -> None:
    """
    Valide la configuration d'entraînement avant le lancement.

    Args:
        data_yaml: chemin du fichier de configuration YOLO.
        model_weights: nom ou chemin des poids YOLOv8.

    Raises:
        FileNotFoundError: si le fichier de configuration est absent.
        ImportError: si ultralytics n'est pas installé.
    """
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Fichier de configuration YOLO introuvable: {data_yaml}\n"
            "Copiez et adaptez train/yolo_train.yaml."
        )
    if not _YOLO_AVAILABLE:
        raise ImportError(
            "Le module ultralytics est requis pour l'entraînement.\n"
            "Installez-le avec : pip install ultralytics"
        )


def _save_run_config(output_dir: Path, config: Dict[str, Any]) -> None:
    """Sauvegarde les hyperparamètres dans un fichier JSON reproductible."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Configuration d'entraînement sauvegardée: %s", config_path)


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------

def train(
    data_yaml: str = "train/yolo_train.yaml",
    model_weights: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/detect",
    name: str = "train",
    device: Optional[str] = None,
    patience: int = 50,
    workers: int = 4,
    optimizer: str = "auto",
    lr0: float = 0.01,
    augment: bool = True,
) -> Dict[str, Any]:
    """
    Lance l'entraînement YOLOv8 avec les paramètres spécifiés.

    Args:
        data_yaml: chemin du fichier de configuration YOLO (data.yaml).
        model_weights: poids de départ (ex: 'yolov8n.pt', 'yolov8s.pt').
        epochs: nombre d'époques.
        imgsz: taille des images (px).
        batch: taille de lot (-1 = auto).
        project: dossier de sortie parent.
        name: nom du run (sous-dossier dans project/).
        device: 'cpu', 'cuda', '0', '1' … None = auto.
        patience: epochs sans amélioration avant early stopping.
        workers: nombre de workers pour le DataLoader.
        optimizer: optimiseur ('auto', 'SGD', 'Adam', 'AdamW').
        lr0: learning rate initial.
        augment: activer les augmentations de données.

    Returns:
        dict avec les métriques de validation du meilleur modèle.

    Raises:
        FileNotFoundError: si data_yaml est absent.
        ImportError: si ultralytics n'est pas installé.
    """
    data_path = Path(data_yaml)
    validate_config(data_path, model_weights)

    # Résolution du device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    config = {
        "data": str(data_path.resolve()),
        "model": model_weights,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": project,
        "name": name,
        "patience": patience,
        "workers": workers,
        "optimizer": optimizer,
        "lr0": lr0,
        "augment": augment,
    }

    output_dir = Path(project) / name
    _save_run_config(output_dir, config)

    logger.info("Démarrage de l'entraînement YOLOv8...")
    logger.info("  Modèle      : %s", model_weights)
    logger.info("  Data        : %s", data_yaml)
    logger.info("  Époques     : %d", epochs)
    logger.info("  Taille img  : %d", imgsz)
    logger.info("  Device      : %s", device)

    model = YOLO(model_weights)
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        device=device,
        patience=patience,
        workers=workers,
        optimizer=optimizer,
        lr0=lr0,
        augment=augment,
    )

    # Extraire les métriques du meilleur modèle
    metrics: Dict[str, Any] = {}
    if hasattr(results, "results_dict"):
        metrics = {k: float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))}

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Entraînement terminé. Métriques: %s", metrics_path)
    return metrics


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_path: str,
    data_yaml: str = "train/yolo_train.yaml",
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Évalue un modèle YOLOv8 sur le jeu de validation.

    Args:
        model_path: chemin vers les poids du modèle (.pt).
        data_yaml: configuration YOLO avec le chemin du dataset.
        imgsz: taille des images.
        batch: taille de lot.
        device: périphérique ('cpu', 'cuda', …).

    Returns:
        dict: métriques de validation (mAP50, mAP50-95, précision, rappel).
    """
    if not _YOLO_AVAILABLE:
        raise ImportError("ultralytics est requis. Installez avec : pip install ultralytics")

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    model = YOLO(model_path)
    metrics_obj = model.val(data=data_yaml, imgsz=imgsz, batch=batch, device=device)
    metrics: Dict[str, Any] = {}
    if hasattr(metrics_obj, "results_dict"):
        metrics = {k: float(v) for k, v in metrics_obj.results_dict.items() if isinstance(v, (int, float))}
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraînement YOLOv8 (wrapper Python)")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Lancer l'entraînement.")
    p_train.add_argument("--data", default="train/yolo_train.yaml")
    p_train.add_argument("--model", default="yolov8n.pt")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--project", default="runs/detect")
    p_train.add_argument("--name", default="train")
    p_train.add_argument("--device", default=None)
    p_train.add_argument("--patience", type=int, default=50)
    p_train.add_argument("--lr0", type=float, default=0.01)
    p_train.add_argument("--no-augment", dest="augment", action="store_false", default=True)

    # ── val ────────────────────────────────────────────────────────────────
    p_val = sub.add_parser("val", help="Évaluer un modèle sur le jeu de validation.")
    p_val.add_argument("--model", required=True, help="Chemin du modèle .pt")
    p_val.add_argument("--data", default="train/yolo_train.yaml")
    p_val.add_argument("--imgsz", type=int, default=640)
    p_val.add_argument("--batch", type=int, default=16)
    p_val.add_argument("--device", default=None)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    if args.command == "train":
        metrics = train(
            data_yaml=args.data,
            model_weights=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
            device=args.device,
            patience=args.patience,
            lr0=args.lr0,
            augment=args.augment,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    elif args.command == "val":
        metrics = evaluate(
            model_path=args.model,
            data_yaml=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
