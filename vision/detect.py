# Fichier: vision/detect.py — Détection locale d'ingrédients/objets culinaires via YOLOv8 (CPU ou GPU).
"""
Ce module propose une interface simple pour charger un modèle YOLOv8 (ultralytics) et
exécuter une détection d'objets sur une image. Il privilégie le GPU si disponible,
retombe automatiquement sur le CPU sinon, et retourne un JSON standardisé pour
faciliter l'intégration avec l'API FastAPI.

Fonctions principales
---------------------
- select_device() -> str : choisit automatiquement entre 'cuda' et 'cpu'.
- load_model(model_name: str, device: str | None) -> YOLO : charge le modèle.
- run_inference(image_path: str, model_name: str, conf: float) -> dict : exécute
  la détection et retourne un dictionnaire sérialisable en JSON.
- save_csv(detections: list[dict], csv_path: str) : exporte les boîtes prédites.

Exemple d'utilisation
---------------------
>>> python vision/detect.py --image data/example.jpg --conf 0.25 --save-json sorties.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - dépendance externe
    YOLO = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:
    import torch
except Exception:  # pragma: no cover - torch est requis mais on garde un fallback lisible
    torch = None


def select_device() -> str:
    """
    Sélectionne automatiquement le périphérique d'exécution.

    Returns:
        str: 'cuda' si un GPU CUDA est disponible et torch importable, sinon 'cpu'.
    """
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name: str = "yolov8n.pt", device: str | None = None) -> Any:
    """
    Charge un modèle YOLOv8 pré-entraîné.

    Args:
        model_name: nom ou chemin vers le poids YOLOv8 (ex: 'yolov8n.pt').
        device: périphérique forcé ('cpu' ou 'cuda'). Si None, sélection automatique.

    Returns:
        Instance du modèle YOLO prêt pour l'inférence.

    Raises:
        ImportError: si ultralytics n'est pas installé.
        RuntimeError: si le modèle ne peut pas être chargé.
    """
    if YOLO is None:
        raise ImportError(
            f"Le module ultralytics est requis mais n'a pas pu être importé : {_IMPORT_ERROR}"
        )
    chosen_device = device or select_device()
    try:
        model = YOLO(model_name)
        model.to(chosen_device)
        return model
    except Exception as exc:  # pragma: no cover - dépendances externes
        raise RuntimeError(f"Échec du chargement du modèle {model_name}: {exc}") from exc


def run_inference(
    image_path: str,
    model_name: str = "yolov8n.pt",
    conf: float = 0.25,
) -> Dict[str, Any]:
    """
    Lance une inférence sur une image et produit un dictionnaire JSON-serializable.

    Args:
        image_path: chemin de l'image à analyser.
        model_name: poids YOLOv8 à utiliser.
        conf: seuil de confiance minimal pour conserver une détection.

    Returns:
        dict: métadonnées et liste des détections (boîtes, classes, scores).
    """
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Image introuvable: {image_path}")

    model = load_model(model_name=model_name)
    results = model(image_path_obj, conf=conf)
    parsed: List[Dict[str, Any]] = []
    for box in results[0].boxes:
        cls_idx = int(box.cls.item())
        score = float(box.conf.item())
        xyxy = box.xyxy.cpu().numpy().tolist()[0]
        names = results[0].names if hasattr(results[0], "names") else model.model.names
        parsed.append(
            {
                "class_id": cls_idx,
                "class_name": names.get(cls_idx, str(cls_idx)),
                "confidence": score,
                "bbox_xyxy": xyxy,
            }
        )

    return {
        "image": str(image_path_obj),
        "model": model_name,
        "device": select_device(),
        "conf_threshold": conf,
        "detections": parsed,
    }


def save_json(payload: Dict[str, Any], json_path: str) -> None:
    """
    Sauvegarde le dictionnaire de résultats au format JSON.

    Args:
        payload: dictionnaire renvoyé par run_inference.
        json_path: chemin de sortie.
    """
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_csv(detections: List[Dict[str, Any]], csv_path: str) -> None:
    """
    Export des détections au format CSV (une ligne par boîte).

    Args:
        detections: liste de dictionnaires décrivant chaque boîte.
        csv_path: chemin du fichier CSV de sortie.
    """
    import csv

    headers = ["class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            writer.writerow(
                {
                    "class_id": det["class_id"],
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Analyse les arguments CLI.

    Args:
        argv: liste des arguments (None pour utiliser sys.argv).

    Returns:
        argparse.Namespace: arguments parsés.
    """
    parser = argparse.ArgumentParser(description="Détection d'ingrédients via YOLOv8")
    parser.add_argument("--image", required=True, help="Chemin vers l'image à analyser.")
    parser.add_argument("--model", default="yolov8n.pt", help="Poids YOLOv8 à utiliser.")
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance.")
    parser.add_argument("--save-json", help="Chemin pour sauvegarder le JSON.")
    parser.add_argument("--save-csv", help="Chemin pour sauvegarder les résultats CSV.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Point d'entrée CLI : charge le modèle, exécute l'inférence et exporte les résultats.
    """
    args = parse_args(argv)
    results = run_inference(args.image, model_name=args.model, conf=args.conf)
    if args.save_json:
        save_json(results, args.save_json)
        print(f"JSON sauvegardé dans {args.save_json}")
    if args.save_csv:
        save_csv(results["detections"], args.save_csv)
        print(f"CSV sauvegardé dans {args.save_csv}")
    if not (args.save_json or args.save_csv):
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
