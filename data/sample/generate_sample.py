#!/usr/bin/env python
# Fichier: data/sample/generate_sample.py — Génère des données d'exemple pour expérimenter.
"""
Ce script crée dans data/sample/ :
- sample.jpg         : une image synthétique 320×240 avec des formes colorées.
- annotations_coco.json : annotations COCO minimales pour sample.jpg.
- annotations.csv    : annotations au format CSV (file,xmin,ymin,xmax,ymax,class_name).

Usage
-----
python data/sample/generate_sample.py
# ou via Makefile :
make data-sample
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


SAMPLE_DIR = Path(__file__).parent
IMG_W, IMG_H = 320, 240

# Boîtes englobantes (xmin, ymin, xmax, ymax) pour chaque ingrédient
TOMATO_BBOX   = (30, 50, 110, 130)
CARROT_BBOX   = (140, 80, 200, 180)
BROCCOLI_BBOX = (230, 40, 300, 110)


def create_sample_image(path: Path) -> None:
    """Crée une image JPEG synthétique avec 3 formes représentant des ingrédients."""
    img = Image.new("RGB", (IMG_W, IMG_H), color=(240, 235, 220))
    draw = ImageDraw.Draw(img)

    # Tomate (rouge, ellipse)
    draw.ellipse(list(TOMATO_BBOX), fill=(210, 50, 50), outline=(150, 30, 30), width=2)

    # Carotte (orange, rectangle arrondi approché)
    draw.rounded_rectangle(list(CARROT_BBOX), radius=10, fill=(230, 130, 40), outline=(180, 90, 20), width=2)

    # Brocoli (vert, cercle)
    draw.ellipse(list(BROCCOLI_BBOX), fill=(60, 160, 60), outline=(30, 100, 30), width=2)

    img.save(path, format="JPEG", quality=90)
    print(f"[OK] Image synthétique créée : {path}")


def create_coco_annotations(path: Path) -> None:
    """Crée un fichier JSON COCO minimal pour sample.jpg."""
    # Conversion (xmin, ymin, xmax, ymax) → (xmin, ymin, w, h) pour COCO
    def to_coco(bbox: tuple) -> list:
        xmin, ymin, xmax, ymax = bbox
        return [xmin, ymin, xmax - xmin, ymax - ymin]

    coco = {
        "images": [
            {"id": 1, "file_name": "sample.jpg", "width": IMG_W, "height": IMG_H}
        ],
        "categories": [
            {"id": 1, "name": "tomate"},
            {"id": 2, "name": "carotte"},
            {"id": 3, "name": "brocoli"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": to_coco(TOMATO_BBOX),   "area": 6400},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": to_coco(CARROT_BBOX),   "area": 6000},
            {"id": 3, "image_id": 1, "category_id": 3, "bbox": to_coco(BROCCOLI_BBOX), "area": 4900},
        ],
    }
    path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Annotations COCO créées   : {path}")


def create_csv_annotations(path: Path) -> None:
    """Crée un CSV d'annotations au format attendu par data/convert_to_yolo.py."""
    def row(fname: str, bbox: tuple, cls: str) -> str:
        xmin, ymin, xmax, ymax = bbox
        return f"{fname},{xmin},{ymin},{xmax},{ymax},{cls}"

    rows = [
        "file,xmin,ymin,xmax,ymax,class_name",
        row("sample.jpg", TOMATO_BBOX,   "tomate"),
        row("sample.jpg", CARROT_BBOX,   "carotte"),
        row("sample.jpg", BROCCOLI_BBOX, "brocoli"),
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[OK] Annotations CSV créées    : {path}")


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    create_sample_image(SAMPLE_DIR / "sample.jpg")
    create_coco_annotations(SAMPLE_DIR / "annotations_coco.json")
    create_csv_annotations(SAMPLE_DIR / "annotations.csv")
    print(
        "\nExpérimentation rapide :"
        "\n  Conversion COCO → YOLO :"
        "\n    python data/convert_to_yolo.py --coco data/sample/annotations_coco.json --output-dir data/sample/yolo_out"
        "\n"
        "\n  Conversion CSV → YOLO :"
        "\n    python data/convert_to_yolo.py --csv data/sample/annotations.csv --images-dir data/sample --output-dir data/sample/yolo_out"
        "\n"
        "\n  Détection sur l'image (nécessite ultralytics) :"
        "\n    python vision/detect.py --image data/sample/sample.jpg --conf 0.25"
    )


if __name__ == "__main__":
    main()
