# Fichier: tests/test_detect.py — Tests unitaires pour vision/detect.py.
"""
Ces tests vérifient le comportement des fonctions utilitaires de vision/detect.py
sans charger de modèle ML réel (pas de dépendance ultralytics/torch requise).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import FAKE_DETECTIONS, FAKE_INFERENCE_RESULT


# ---------------------------------------------------------------------------
# select_device
# ---------------------------------------------------------------------------

class TestSelectDevice:
    def test_returns_cpu_when_torch_unavailable(self):
        """Sans torch, select_device doit renvoyer 'cpu'."""
        import vision.detect as det_mod
        original = det_mod.torch
        det_mod.torch = None
        try:
            assert det_mod.select_device() == "cpu"
        finally:
            det_mod.torch = original

    def test_returns_cpu_when_cuda_unavailable(self):
        """Quand CUDA n'est pas disponible, select_device renvoie 'cpu'."""
        import vision.detect as det_mod
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.object(det_mod, "torch", mock_torch):
            assert det_mod.select_device() == "cpu"

    def test_returns_cuda_when_available(self):
        """Quand CUDA est disponible, select_device renvoie 'cuda'."""
        import vision.detect as det_mod
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.object(det_mod, "torch", mock_torch):
            assert det_mod.select_device() == "cuda"

    def test_return_type_is_string(self):
        """select_device retourne toujours une chaîne."""
        from vision.detect import select_device
        assert isinstance(select_device(), str)


# ---------------------------------------------------------------------------
# save_json
# ---------------------------------------------------------------------------

class TestSaveJson:
    def test_creates_file(self, tmp_path: Path):
        """save_json crée le fichier JSON."""
        from vision.detect import save_json
        out = tmp_path / "result.json"
        save_json(FAKE_INFERENCE_RESULT, str(out))
        assert out.exists()

    def test_content_is_valid_json(self, tmp_path: Path):
        """Le fichier produit doit être un JSON valide."""
        from vision.detect import save_json
        out = tmp_path / "result.json"
        save_json(FAKE_INFERENCE_RESULT, str(out))
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["model"] == "yolov8n.pt"
        assert data["device"] == "cpu"

    def test_detections_preserved(self, tmp_path: Path):
        """Toutes les détections doivent être dans le JSON exporté."""
        from vision.detect import save_json
        out = tmp_path / "result.json"
        save_json(FAKE_INFERENCE_RESULT, str(out))
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["detections"]) == len(FAKE_DETECTIONS)
        assert data["detections"][0]["class_name"] == "tomate"


# ---------------------------------------------------------------------------
# save_csv
# ---------------------------------------------------------------------------

class TestSaveCsv:
    def test_creates_file(self, tmp_path: Path):
        """save_csv crée le fichier CSV."""
        from vision.detect import save_csv
        out = tmp_path / "detections.csv"
        save_csv(FAKE_DETECTIONS, str(out))
        assert out.exists()

    def test_header_row(self, tmp_path: Path):
        """Le CSV doit commencer par un en-tête correct."""
        from vision.detect import save_csv
        out = tmp_path / "detections.csv"
        save_csv(FAKE_DETECTIONS, str(out))
        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "class_name" in header
        assert "confidence" in header

    def test_row_count(self, tmp_path: Path):
        """Le nombre de lignes de données doit correspondre aux détections."""
        from vision.detect import save_csv
        out = tmp_path / "detections.csv"
        save_csv(FAKE_DETECTIONS, str(out))
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(FAKE_DETECTIONS)

    def test_empty_detections(self, tmp_path: Path):
        """save_csv ne doit pas lever d'exception avec une liste vide."""
        from vision.detect import save_csv
        out = tmp_path / "empty.csv"
        save_csv([], str(out))
        assert out.exists()

    def test_bbox_columns_present(self, tmp_path: Path):
        """Les colonnes x1/y1/x2/y2 doivent être présentes et bien remplies."""
        from vision.detect import save_csv
        out = tmp_path / "detections.csv"
        save_csv(FAKE_DETECTIONS, str(out))
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["x1"]) == pytest.approx(10.0)
        assert float(rows[0]["y2"]) == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# run_inference
# ---------------------------------------------------------------------------

class TestRunInference:
    def test_raises_file_not_found(self, tmp_path: Path):
        """run_inference lève FileNotFoundError si l'image est absente."""
        from vision.detect import run_inference
        with pytest.raises(FileNotFoundError):
            run_inference(str(tmp_path / "ghost.jpg"))

    def test_calls_model_with_conf(self, synthetic_image: Path):
        """run_inference passe le bon seuil de confiance au modèle mocké."""
        import vision.detect as det_mod

        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_result.names = {0: "tomate"}
        mock_model.return_value = [mock_result]
        mock_model.model.names = {0: "tomate"}

        with patch.object(det_mod, "load_model", return_value=mock_model):
            result = det_mod.run_inference(str(synthetic_image), conf=0.5)

        mock_model.assert_called_once_with(synthetic_image, conf=0.5)
        assert result["conf_threshold"] == pytest.approx(0.5)

    def test_result_structure(self, synthetic_image: Path):
        """Le résultat de run_inference doit contenir toutes les clés attendues."""
        import vision.detect as det_mod

        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_result.names = {0: "ingredient"}
        mock_model.return_value = [mock_result]

        with patch.object(det_mod, "load_model", return_value=mock_model):
            result = det_mod.run_inference(str(synthetic_image))

        for key in ("image", "model", "device", "conf_threshold", "detections"):
            assert key in result

    def test_detections_list_format(self, synthetic_image: Path):
        """Chaque détection doit contenir class_id, class_name, confidence, bbox_xyxy."""
        import vision.detect as det_mod

        # Construire un box mocké réaliste
        mock_box = MagicMock()
        mock_box.cls.item.return_value = 0
        mock_box.conf.item.return_value = 0.85
        mock_box.xyxy.cpu.return_value.numpy.return_value.tolist.return_value = [
            [5.0, 10.0, 30.0, 40.0]
        ]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "pomme"}

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]

        with patch.object(det_mod, "load_model", return_value=mock_model):
            result = det_mod.run_inference(str(synthetic_image))

        assert len(result["detections"]) == 1
        det = result["detections"][0]
        assert det["class_id"] == 0
        assert det["class_name"] == "pomme"
        assert det["confidence"] == pytest.approx(0.85)
        assert det["bbox_xyxy"] == [5.0, 10.0, 30.0, 40.0]


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_raises_import_error_when_yolo_none(self):
        """load_model doit lever ImportError si YOLO n'est pas importable."""
        import vision.detect as det_mod
        original_yolo = det_mod.YOLO
        det_mod.YOLO = None
        try:
            with pytest.raises(ImportError):
                det_mod.load_model()
        finally:
            det_mod.YOLO = original_yolo


# ---------------------------------------------------------------------------
# get_class_names
# ---------------------------------------------------------------------------

class TestGetClassNames:
    def test_uses_result_names(self):
        """get_class_names préfère les noms du résultat."""
        from vision.detect import get_class_names
        result = MagicMock()
        result.names = {0: "fromage", 1: "oeuf"}
        names = get_class_names(result, MagicMock())
        assert names[0] == "fromage"

    def test_fallback_to_model_names(self):
        """get_class_names utilise model.model.names si result.names est absent."""
        from vision.detect import get_class_names
        result = MagicMock()
        result.names = None
        model = MagicMock()
        model.model.names = {0: "pain"}
        names = get_class_names(result, model)
        assert names[0] == "pain"

    def test_raises_when_no_names(self):
        """get_class_names lève RuntimeError si aucun mapping n'est disponible."""
        from vision.detect import get_class_names
        result = MagicMock()
        result.names = None
        model = MagicMock()
        model.model.names = None
        with pytest.raises(RuntimeError):
            get_class_names(result, model)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_required_image_argument(self):
        """--image est obligatoire."""
        from vision.detect import parse_args
        args = parse_args(["--image", "photo.jpg"])
        assert args.image == "photo.jpg"

    def test_default_conf(self):
        """La confiance par défaut est 0.25."""
        from vision.detect import parse_args
        args = parse_args(["--image", "photo.jpg"])
        assert args.conf == pytest.approx(0.25)

    def test_default_model(self):
        """Le modèle par défaut est yolov8n.pt."""
        from vision.detect import parse_args
        args = parse_args(["--image", "photo.jpg"])
        assert args.model == "yolov8n.pt"

    def test_custom_conf(self):
        """--conf accepte une valeur personnalisée."""
        from vision.detect import parse_args
        args = parse_args(["--image", "p.jpg", "--conf", "0.6"])
        assert args.conf == pytest.approx(0.6)

    def test_save_json_and_csv_args(self):
        """--save-json et --save-csv sont correctement parsés."""
        from vision.detect import parse_args
        args = parse_args(
            ["--image", "p.jpg", "--save-json", "out.json", "--save-csv", "out.csv"]
        )
        assert args.save_json == "out.json"
        assert args.save_csv == "out.csv"

    def test_missing_image_raises(self):
        """Omettre --image doit lever SystemExit."""
        from vision.detect import parse_args
        with pytest.raises(SystemExit):
            parse_args([])
