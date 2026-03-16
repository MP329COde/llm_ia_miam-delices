# Fichier: tests/conftest.py — Fixtures partagées pour tous les tests.
"""
Fixtures pytest réutilisables : images synthétiques, JSON COCO minimal, CSV
d'annotations. Ces fichiers temporaires sont créés et détruits automatiquement
pour chaque test qui les utilise.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Images synthétiques
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_image(tmp_path: Path) -> Path:
    """Crée une petite image JPEG synthétique (32×32, unie rouge)."""
    img_path = tmp_path / "test_image.jpg"
    img = Image.new("RGB", (32, 32), color=(220, 50, 50))
    img.save(img_path, format="JPEG")
    return img_path


@pytest.fixture()
def synthetic_png(tmp_path: Path) -> Path:
    """Crée une petite image PNG synthétique (64×64, unie verte)."""
    img_path = tmp_path / "test_image.png"
    img = Image.new("RGB", (64, 64), color=(50, 200, 50))
    img.save(img_path, format="PNG")
    return img_path


# ---------------------------------------------------------------------------
# Données COCO minimales
# ---------------------------------------------------------------------------

@pytest.fixture()
def coco_json_file(tmp_path: Path) -> Path:
    """
    Crée un fichier JSON COCO minimal avec 2 images, 2 catégories et 3 annotations.
    """
    coco_data = {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "img2.jpg", "width": 200, "height": 150},
        ],
        "categories": [
            {"id": 1, "name": "tomate"},
            {"id": 2, "name": "carotte"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [5, 5, 10, 10]},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [50, 60, 80, 70]},
        ],
    }
    coco_file = tmp_path / "annotations.json"
    coco_file.write_text(json.dumps(coco_data), encoding="utf-8")
    return coco_file


# ---------------------------------------------------------------------------
# Données CSV minimales
# ---------------------------------------------------------------------------

@pytest.fixture()
def csv_annotation_dir(tmp_path: Path) -> tuple[Path, Path]:
    """
    Crée un répertoire d'images synthétiques et un CSV d'annotations associé.

    Returns:
        (csv_path, images_dir)
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Créer les images correspondantes
    for name, color in [("apple.jpg", (200, 50, 50)), ("banana.jpg", (230, 210, 50))]:
        img = Image.new("RGB", (80, 60), color=color)
        img.save(images_dir / name, format="JPEG")

    csv_content = textwrap.dedent("""\
        file,xmin,ymin,xmax,ymax,class_name
        apple.jpg,5,5,40,30,pomme
        apple.jpg,45,10,75,55,pomme
        banana.jpg,10,10,70,50,banane
    """)
    csv_file = tmp_path / "boxes.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    return csv_file, images_dir


# ---------------------------------------------------------------------------
# Détections fictives (sans modèle ML)
# ---------------------------------------------------------------------------

FAKE_DETECTIONS = [
    {
        "class_id": 0,
        "class_name": "tomate",
        "confidence": 0.92,
        "bbox_xyxy": [10.0, 20.0, 50.0, 60.0],
    },
    {
        "class_id": 1,
        "class_name": "carotte",
        "confidence": 0.75,
        "bbox_xyxy": [70.0, 80.0, 120.0, 130.0],
    },
]

FAKE_INFERENCE_RESULT = {
    "image": "data/sample/test_image.jpg",
    "model": "yolov8n.pt",
    "device": "cpu",
    "conf_threshold": 0.25,
    "detections": FAKE_DETECTIONS,
}
