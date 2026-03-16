# Fichier: data/convert_to_yolo.py — Conversion d'annotations COCO ou CSV vers le format YOLO.
"""
Ce script convertit des annotations COCO ou un CSV simple vers le format YOLO
(fichiers .txt avec `class cx cy w h` normalisés entre 0 et 1). Il crée la
hiérarchie `output_dir/labels/` et réutilise les noms d'images existants.

Fonctions principales
---------------------
- convert_coco_json_to_yolo(coco_json, output_dir): gère le format COCO officiel.
- convert_csv_to_yolo(csv_path, images_dir, output_dir): gère un CSV avec colonnes
  file,xmin,ymin,xmax,ymax,class_name.

Exemple COCO
------------
>>> python data/convert_to_yolo.py --coco annotations.json --output-dir yolo_out

Exemple CSV
-----------
>>> python data/convert_to_yolo.py --csv boxes.csv --images-dir images/ --output-dir yolo_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from PIL import Image


def coco_bbox_to_yolo(bbox: Iterable[float], width: int, height: int) -> Tuple[float, float, float, float]:
    """
    Convertit une boîte COCO (x_min, y_min, w, h) en coordonnées YOLO normalisées.

    Args:
        bbox: liste/tuple [x_min, y_min, w, h].
        width: largeur de l'image.
        height: hauteur de l'image.

    Returns:
        tuple: (cx, cy, w, h) normalisés entre 0 et 1.
    """
    x_min, y_min, w, h = bbox
    cx = (x_min + w / 2) / width
    cy = (y_min + h / 2) / height
    return cx, cy, w / width, h / height


def ensure_dir(path: Path) -> None:
    """
    Crée un répertoire s'il n'existe pas.
    """
    path.mkdir(parents=True, exist_ok=True)


def write_yolo_file(output_dir: Path, image_name: str, lines: List[str]) -> None:
    """
    Écrit un fichier .txt YOLO pour une image.

    Args:
        output_dir: dossier racine où sera créé labels/<image>.txt.
        image_name: nom du fichier image (sans extension pour l'étiquette).
        lines: lignes au format YOLO à écrire.
    """
    labels_dir = output_dir / "labels"
    ensure_dir(labels_dir)
    out_file = labels_dir / f"{Path(image_name).stem}.txt"
    out_file.write_text("\n".join(lines), encoding="utf-8")


def convert_coco_json_to_yolo(coco_json: Path, output_dir: Path) -> None:
    """
    Convertit un fichier d'annotations COCO en labels YOLO.

    Args:
        coco_json: chemin vers le fichier COCO (.json).
        output_dir: répertoire racine de sortie (créé si besoin).
    """
    data = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    images = {img["id"]: img for img in data.get("images", [])}
    categories = {cat["id"]: idx for idx, cat in enumerate(data.get("categories", []))}

    image_to_lines: Dict[int, List[str]] = {img_id: [] for img_id in images}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in images:
            continue
        img_info = images[img_id]
        width, height = img_info["width"], img_info["height"]
        cx, cy, w, h = coco_bbox_to_yolo(ann["bbox"], width, height)
        class_id = categories.get(ann["category_id"], ann["category_id"])
        line = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        image_to_lines[img_id].append(line)

    for img_id, lines in image_to_lines.items():
        img_name = images[img_id]["file_name"]
        write_yolo_file(output_dir, img_name, lines)


def convert_csv_to_yolo(csv_path: Path, images_dir: Path, output_dir: Path) -> None:
    """
    Convertit un CSV simple en labels YOLO. Le CSV doit contenir:
    file,xmin,ymin,xmax,ymax,class_name

    Args:
        csv_path: chemin du CSV d'annotations.
        images_dir: dossier contenant les images référencées.
        output_dir: dossier de sortie YOLO.
    """
    df = pd.read_csv(csv_path)
    required = {"file", "xmin", "ymin", "xmax", "ymax", "class_name"}
    if not required.issubset(df.columns):
        raise ValueError(f"Colonnes attendues: {required}")

    class_to_id: Dict[str, int] = {}
    image_to_lines: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        image_file = str(row["file"])
        img_path = images_dir / image_file
        if not img_path.exists():
            raise FileNotFoundError(f"Image manquante: {img_path}")
        with Image.open(img_path) as img:
            width, height = img.size

        class_name = str(row["class_name"])
        if class_name not in class_to_id:
            class_to_id[class_name] = len(class_to_id)
        class_id = class_to_id[class_name]

        x_min = float(row["xmin"])
        y_min = float(row["ymin"])
        x_max = float(row["xmax"])
        y_max = float(row["ymax"])
        w, h = x_max - x_min, y_max - y_min
        cx, cy, w_norm, h_norm = coco_bbox_to_yolo((x_min, y_min, w, h), width, height)
        line = f"{class_id} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}"

        image_to_lines.setdefault(image_file, []).append(line)

    for image_name, lines in image_to_lines.items():
        write_yolo_file(output_dir, image_name, lines)

    # Sauvegarde du mapping des classes pour référence
    mapping_path = output_dir / "classes.txt"
    mapping_lines = [f"{idx},{name}" for name, idx in class_to_id.items()]
    mapping_path.write_text("\n".join(mapping_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """
    Parse les arguments CLI.
    """
    parser = argparse.ArgumentParser(description="Convertir COCO/CSV vers YOLO.")
    parser.add_argument("--coco", type=Path, help="Fichier d'annotations COCO (.json).")
    parser.add_argument("--csv", type=Path, help="Fichier CSV (file,xmin,ymin,xmax,ymax,class_name).")
    parser.add_argument("--images-dir", type=Path, help="Dossier contenant les images (requis pour CSV).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Dossier de sortie YOLO.")
    return parser.parse_args()


def main() -> None:
    """
    Point d'entrée CLI. Déclenche la conversion demandée.
    """
    args = parse_args()
    output_dir = args.output_dir
    ensure_dir(output_dir / "labels")

    if args.coco:
        convert_coco_json_to_yolo(args.coco, output_dir)
        print(f"Conversion COCO terminée -> {output_dir}")
    if args.csv:
        if args.images_dir is None:
            raise SystemExit("--images-dir est requis pour convertir un CSV")
        convert_csv_to_yolo(args.csv, args.images_dir, output_dir)
        print(f"Conversion CSV terminée -> {output_dir}")
    if not (args.coco or args.csv):
        raise SystemExit("Spécifiez --coco ou --csv")


if __name__ == "__main__":
    main()
