# Fichier: tests/test_train.py — Tests pour train/train.py.
"""
Tests du wrapper Python d'entraînement YOLOv8.
Les appels réels à ultralytics sont mockés.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

class TestValidateConfig:
    def test_raises_file_not_found_if_yaml_missing(self, tmp_path: Path):
        from train.train import validate_config
        with pytest.raises(FileNotFoundError):
            validate_config(tmp_path / "missing.yaml", "yolov8n.pt")

    def test_raises_import_error_if_yolo_unavailable(self, tmp_path: Path):
        from train.train import validate_config
        yaml = tmp_path / "data.yaml"
        yaml.write_text("path: .\n", encoding="utf-8")
        import train.train as train_mod
        with patch.object(train_mod, "_YOLO_AVAILABLE", False):
            with pytest.raises(ImportError):
                validate_config(yaml, "yolov8n.pt")

    def test_no_error_when_valid(self, tmp_path: Path):
        from train.train import validate_config
        yaml = tmp_path / "data.yaml"
        yaml.write_text("path: .\n", encoding="utf-8")
        import train.train as train_mod
        with patch.object(train_mod, "_YOLO_AVAILABLE", True):
            validate_config(yaml, "yolov8n.pt")  # Ne doit pas lever d'exception


# ---------------------------------------------------------------------------
# _save_run_config
# ---------------------------------------------------------------------------

class TestSaveRunConfig:
    def test_creates_json_file(self, tmp_path: Path):
        from train.train import _save_run_config
        config = {"epochs": 10, "model": "yolov8n.pt"}
        out_dir = tmp_path / "run"
        _save_run_config(out_dir, config)
        assert (out_dir / "run_config.json").exists()

    def test_json_content(self, tmp_path: Path):
        from train.train import _save_run_config
        import json
        config = {"epochs": 10, "batch": 8}
        out_dir = tmp_path / "run"
        _save_run_config(out_dir, config)
        data = json.loads((out_dir / "run_config.json").read_text())
        assert data["epochs"] == 10
        assert data["batch"] == 8


# ---------------------------------------------------------------------------
# train (mocked)
# ---------------------------------------------------------------------------

class TestTrain:
    def test_raises_if_yaml_missing(self, tmp_path: Path):
        from train.train import train
        with pytest.raises(FileNotFoundError):
            train(data_yaml=str(tmp_path / "missing.yaml"))

    def test_calls_yolo_train(self, tmp_path: Path):
        """Vérifie que train() appelle model.train() avec les bons paramètres."""
        from train.train import train
        import train.train as train_mod

        yaml = tmp_path / "data.yaml"
        yaml.write_text("path: .\n", encoding="utf-8")

        mock_results = MagicMock()
        mock_results.results_dict = {"metrics/mAP50": 0.85}
        mock_model = MagicMock()
        mock_model.train.return_value = mock_results
        mock_yolo_cls = MagicMock(return_value=mock_model)

        with patch.object(train_mod, "YOLO", mock_yolo_cls), \
             patch.object(train_mod, "_YOLO_AVAILABLE", True):
            metrics = train(
                data_yaml=str(yaml),
                model_weights="yolov8n.pt",
                epochs=5,
                project=str(tmp_path / "runs"),
                name="test_run",
                device="cpu",
            )

        mock_model.train.assert_called_once()
        call_kwargs = mock_model.train.call_args[1]
        assert call_kwargs["epochs"] == 5
        assert call_kwargs["device"] == "cpu"

    def test_returns_metrics_dict(self, tmp_path: Path):
        from train.train import train
        import train.train as train_mod

        yaml = tmp_path / "data.yaml"
        yaml.write_text("path: .\n", encoding="utf-8")

        mock_results = MagicMock()
        mock_results.results_dict = {"metrics/mAP50": 0.75, "metrics/mAP50-95": 0.60}
        mock_model = MagicMock()
        mock_model.train.return_value = mock_results
        mock_yolo_cls = MagicMock(return_value=mock_model)

        with patch.object(train_mod, "YOLO", mock_yolo_cls), \
             patch.object(train_mod, "_YOLO_AVAILABLE", True):
            metrics = train(data_yaml=str(yaml), epochs=1, project=str(tmp_path), name="r")

        assert isinstance(metrics, dict)

    def test_saves_run_config(self, tmp_path: Path):
        from train.train import train
        import train.train as train_mod
        import json

        yaml = tmp_path / "data.yaml"
        yaml.write_text("path: .\n", encoding="utf-8")
        project = str(tmp_path / "runs")

        mock_results = MagicMock()
        mock_results.results_dict = {}
        mock_model = MagicMock()
        mock_model.train.return_value = mock_results
        mock_yolo_cls = MagicMock(return_value=mock_model)

        with patch.object(train_mod, "YOLO", mock_yolo_cls), \
             patch.object(train_mod, "_YOLO_AVAILABLE", True):
            train(data_yaml=str(yaml), epochs=1, project=project, name="myrun")

        cfg_path = Path(project) / "myrun" / "run_config.json"
        assert cfg_path.exists()
        cfg = json.loads(cfg_path.read_text())
        assert cfg["epochs"] == 1


# ---------------------------------------------------------------------------
# evaluate (mocked)
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_raises_import_error_if_unavailable(self):
        from train.train import evaluate
        import train.train as train_mod
        with patch.object(train_mod, "_YOLO_AVAILABLE", False):
            with pytest.raises(ImportError):
                evaluate("model.pt")

    def test_returns_dict(self, tmp_path: Path):
        from train.train import evaluate
        import train.train as train_mod

        mock_metrics = MagicMock()
        mock_metrics.results_dict = {"metrics/mAP50": 0.8}
        mock_model = MagicMock()
        mock_model.val.return_value = mock_metrics
        mock_yolo_cls = MagicMock(return_value=mock_model)

        with patch.object(train_mod, "YOLO", mock_yolo_cls), \
             patch.object(train_mod, "_YOLO_AVAILABLE", True):
            metrics = evaluate("fake_model.pt", data_yaml="fake.yaml", device="cpu")

        assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgsTrain:
    def test_train_subcommand(self):
        from train.train import parse_args
        args = parse_args(["train", "--data", "data.yaml", "--epochs", "10"])
        assert args.command == "train"
        assert args.epochs == 10

    def test_val_subcommand(self):
        from train.train import parse_args
        args = parse_args(["val", "--model", "best.pt"])
        assert args.command == "val"
        assert args.model == "best.pt"

    def test_missing_subcommand_exits(self):
        from train.train import parse_args
        with pytest.raises(SystemExit):
            parse_args([])
