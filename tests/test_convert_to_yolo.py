# Fichier: tests/test_convert_to_yolo.py — Tests unitaires pour data/convert_to_yolo.py.
"""
Vérifie les conversions d'annotations COCO et CSV vers le format YOLO.
Toutes les entrées sont générées en mémoire via des fixtures temporaires.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.convert_to_yolo import (
    coco_bbox_to_yolo,
    convert_coco_json_to_yolo,
    convert_csv_to_yolo,
    ensure_dir,
    write_yolo_file,
)


# ---------------------------------------------------------------------------
# coco_bbox_to_yolo
# ---------------------------------------------------------------------------

class TestCocoBboxToYolo:
    def test_center_point(self):
        """Vérifie le calcul du centre normalisé."""
        cx, cy, w, h = coco_bbox_to_yolo([0, 0, 100, 100], 200, 200)
        assert cx == pytest.approx(0.25)
        assert cy == pytest.approx(0.25)

    def test_width_height_normalized(self):
        """w et h doivent être normalisés entre 0 et 1."""
        cx, cy, w, h = coco_bbox_to_yolo([10, 20, 40, 60], 100, 200)
        assert w == pytest.approx(0.40)
        assert h == pytest.approx(0.30)

    def test_full_image_bbox(self):
        """Une boîte couvrant toute l'image doit donner cx=0.5, cy=0.5, w=1, h=1."""
        cx, cy, w, h = coco_bbox_to_yolo([0, 0, 100, 100], 100, 100)
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)
        assert w == pytest.approx(1.0)
        assert h == pytest.approx(1.0)

    def test_small_bbox(self):
        """Une petite boîte en haut à gauche."""
        cx, cy, w, h = coco_bbox_to_yolo([0, 0, 10, 10], 1000, 1000)
        assert cx == pytest.approx(0.005)
        assert w == pytest.approx(0.01)

    def test_returns_four_values(self):
        """La fonction retourne exactement 4 valeurs."""
        result = coco_bbox_to_yolo([0, 0, 50, 50], 100, 100)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# ensure_dir
# ---------------------------------------------------------------------------

class TestEnsureDir:
    def test_creates_missing_directory(self, tmp_path: Path):
        """ensure_dir crée un répertoire inexistant."""
        target = tmp_path / "new" / "nested"
        ensure_dir(target)
        assert target.is_dir()

    def test_no_error_if_exists(self, tmp_path: Path):
        """ensure_dir ne lève pas d'exception si le répertoire existe déjà."""
        ensure_dir(tmp_path)  # already exists


# ---------------------------------------------------------------------------
# write_yolo_file
# ---------------------------------------------------------------------------

class TestWriteYoloFile:
    def test_creates_label_file(self, tmp_path: Path):
        """write_yolo_file crée un fichier dans output_dir/labels/."""
        write_yolo_file(tmp_path, "img.jpg", ["0 0.5 0.5 0.3 0.3"])
        label_file = tmp_path / "labels" / "img.txt"
        assert label_file.exists()

    def test_file_content(self, tmp_path: Path):
        """Le contenu du fichier label doit correspondre aux lignes fournies."""
        lines = ["0 0.5 0.5 0.3 0.3", "1 0.2 0.3 0.1 0.1"]
        write_yolo_file(tmp_path, "photo.png", lines)
        content = (tmp_path / "labels" / "photo.txt").read_text(encoding="utf-8")
        assert content == "\n".join(lines)

    def test_stem_used_not_full_filename(self, tmp_path: Path):
        """Le nom du fichier label utilise le stem (sans extension)."""
        write_yolo_file(tmp_path, "image.jpeg", ["0 0.5 0.5 0.2 0.2"])
        assert (tmp_path / "labels" / "image.txt").exists()


# ---------------------------------------------------------------------------
# convert_coco_json_to_yolo
# ---------------------------------------------------------------------------

class TestConvertCocoJsonToYolo:
    def test_creates_label_files(self, coco_json_file: Path, tmp_path: Path):
        """convert_coco_json_to_yolo crée un fichier label par image."""
        out_dir = tmp_path / "yolo_out"
        convert_coco_json_to_yolo(coco_json_file, out_dir)
        labels_dir = out_dir / "labels"
        assert labels_dir.is_dir()
        assert (labels_dir / "img1.txt").exists()
        assert (labels_dir / "img2.txt").exists()

    def test_annotation_count(self, coco_json_file: Path, tmp_path: Path):
        """img1 a 2 annotations, img2 en a 1."""
        out_dir = tmp_path / "yolo_out"
        convert_coco_json_to_yolo(coco_json_file, out_dir)
        img1_lines = (out_dir / "labels" / "img1.txt").read_text(encoding="utf-8").splitlines()
        img2_lines = (out_dir / "labels" / "img2.txt").read_text(encoding="utf-8").splitlines()
        assert len(img1_lines) == 2
        assert len(img2_lines) == 1

    def test_yolo_format(self, coco_json_file: Path, tmp_path: Path):
        """Chaque ligne doit contenir 5 valeurs numériques."""
        out_dir = tmp_path / "yolo_out"
        convert_coco_json_to_yolo(coco_json_file, out_dir)
        for line in (out_dir / "labels" / "img1.txt").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            assert len(parts) == 5
            float(parts[0])  # class_id
            for p in parts[1:]:
                val = float(p)
                assert 0.0 <= val <= 1.0, f"Valeur hors bornes: {val}"

    def test_empty_annotations(self, tmp_path: Path):
        """Un fichier COCO sans annotations crée des fichiers label vides."""
        coco_data = {
            "images": [{"id": 1, "file_name": "lonely.jpg", "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "chose"}],
            "annotations": [],
        }
        coco_file = tmp_path / "empty.json"
        coco_file.write_text(json.dumps(coco_data), encoding="utf-8")
        out_dir = tmp_path / "yolo_out"
        convert_coco_json_to_yolo(coco_file, out_dir)
        label = out_dir / "labels" / "lonely.txt"
        assert label.exists()
        assert label.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# convert_csv_to_yolo
# ---------------------------------------------------------------------------

class TestConvertCsvToYolo:
    def test_creates_label_files(self, csv_annotation_dir, tmp_path: Path):
        """convert_csv_to_yolo crée un label par image référencée dans le CSV."""
        csv_file, images_dir = csv_annotation_dir
        out_dir = tmp_path / "yolo_out"
        convert_csv_to_yolo(csv_file, images_dir, out_dir)
        assert (out_dir / "labels" / "apple.txt").exists()
        assert (out_dir / "labels" / "banana.txt").exists()

    def test_class_mapping_file(self, csv_annotation_dir, tmp_path: Path):
        """Un fichier classes.txt doit être créé avec le mapping."""
        csv_file, images_dir = csv_annotation_dir
        out_dir = tmp_path / "yolo_out"
        convert_csv_to_yolo(csv_file, images_dir, out_dir)
        assert (out_dir / "classes.txt").exists()
        content = (out_dir / "classes.txt").read_text(encoding="utf-8")
        assert "pomme" in content
        assert "banane" in content

    def test_apple_has_two_annotations(self, csv_annotation_dir, tmp_path: Path):
        """apple.jpg a 2 annotations dans le CSV."""
        csv_file, images_dir = csv_annotation_dir
        out_dir = tmp_path / "yolo_out"
        convert_csv_to_yolo(csv_file, images_dir, out_dir)
        lines = (out_dir / "labels" / "apple.txt").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_yolo_format_csv(self, csv_annotation_dir, tmp_path: Path):
        """Chaque ligne de label doit contenir 5 valeurs numériques."""
        csv_file, images_dir = csv_annotation_dir
        out_dir = tmp_path / "yolo_out"
        convert_csv_to_yolo(csv_file, images_dir, out_dir)
        for line in (out_dir / "labels" / "banana.txt").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            assert len(parts) == 5
            for p in parts[1:]:
                val = float(p)
                assert 0.0 <= val <= 1.0

    def test_missing_columns_raises(self, tmp_path: Path):
        """Un CSV sans les colonnes requises doit lever ValueError."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("file,x1,y1\nimg.jpg,0,0\n", encoding="utf-8")
        with pytest.raises(ValueError):
            convert_csv_to_yolo(bad_csv, tmp_path, tmp_path / "out")

    def test_missing_image_raises(self, tmp_path: Path):
        """Si une image référencée dans le CSV est absente, lever FileNotFoundError."""
        csv_content = "file,xmin,ymin,xmax,ymax,class_name\nghost.jpg,0,0,10,10,thing\n"
        csv_file = tmp_path / "boxes.csv"
        csv_file.write_text(csv_content, encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            convert_csv_to_yolo(csv_file, tmp_path, tmp_path / "out")
