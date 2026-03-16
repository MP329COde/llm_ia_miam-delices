# Fichier: tests/test_caption.py — Tests pour vision/caption.py.
"""
Tous les tests de ce module fonctionnent sans modèle BLIP réel
grâce à des mocks de transformers et PIL.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# is_blip_available
# ---------------------------------------------------------------------------

class TestIsBlipAvailable:
    def test_returns_bool(self):
        from vision.caption import is_blip_available
        assert isinstance(is_blip_available(), bool)

    def test_false_when_blip_not_available(self):
        import vision.caption as cap_mod
        original = cap_mod._BLIP_AVAILABLE
        cap_mod._BLIP_AVAILABLE = False
        try:
            assert not cap_mod.is_blip_available()
        finally:
            cap_mod._BLIP_AVAILABLE = original

    def test_false_when_torch_not_available(self):
        import vision.caption as cap_mod
        original = cap_mod._TORCH_AVAILABLE
        cap_mod._TORCH_AVAILABLE = False
        try:
            assert not cap_mod.is_blip_available()
        finally:
            cap_mod._TORCH_AVAILABLE = original


# ---------------------------------------------------------------------------
# select_device
# ---------------------------------------------------------------------------

class TestSelectDevice:
    def test_returns_cpu_without_cuda(self):
        import vision.caption as cap_mod
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.object(cap_mod, "_torch", mock_torch), \
             patch.object(cap_mod, "_TORCH_AVAILABLE", True):
            assert cap_mod.select_device() == "cpu"

    def test_returns_cuda_with_cuda(self):
        import vision.caption as cap_mod
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.object(cap_mod, "_torch", mock_torch), \
             patch.object(cap_mod, "_TORCH_AVAILABLE", True):
            assert cap_mod.select_device() == "cuda"


# ---------------------------------------------------------------------------
# load_blip
# ---------------------------------------------------------------------------

class TestLoadBlip:
    def test_raises_import_error_when_unavailable(self):
        import vision.caption as cap_mod
        with patch.object(cap_mod, "_BLIP_AVAILABLE", False):
            with pytest.raises(ImportError):
                cap_mod.load_blip()

    def test_returns_processor_and_model(self):
        import vision.caption as cap_mod
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_bp = MagicMock()
        mock_bm = MagicMock()
        mock_bp.from_pretrained.return_value = mock_processor
        mock_bm.from_pretrained.return_value.to.return_value = mock_model
        with patch.object(cap_mod, "_BLIP_AVAILABLE", True), \
             patch.object(cap_mod, "_blip_cache", {}), \
             patch.object(cap_mod, "BlipProcessor", mock_bp), \
             patch.object(cap_mod, "BlipForConditionalGeneration", mock_bm), \
             patch.object(cap_mod, "select_device", return_value="cpu"):
            processor, model = cap_mod.load_blip(model_id="test-model", device="cpu")
            assert processor is mock_processor
            assert model is mock_model

    def test_uses_cache_on_second_call(self):
        import vision.caption as cap_mod
        # Pré-remplir le cache
        fake_cache = {"test-model:cpu": ("proc", "mod")}
        with patch.object(cap_mod, "_blip_cache", fake_cache), \
             patch.object(cap_mod, "_BLIP_AVAILABLE", True):
            result = cap_mod.load_blip(model_id="test-model", device="cpu")
            assert result == ("proc", "mod")


# ---------------------------------------------------------------------------
# generate_caption
# ---------------------------------------------------------------------------

class TestGenerateCaption:
    def test_raises_file_not_found(self, tmp_path: Path):
        from vision.caption import generate_caption
        with pytest.raises(FileNotFoundError):
            generate_caption(str(tmp_path / "ghost.jpg"))

    def test_raises_import_error_when_pil_unavailable(self, synthetic_image: Path):
        import vision.caption as cap_mod
        with patch.object(cap_mod, "_PIL_AVAILABLE", False):
            with pytest.raises(ImportError):
                cap_mod.generate_caption(str(synthetic_image))

    def test_returns_caption_dict(self, synthetic_image: Path):
        import vision.caption as cap_mod
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor = MagicMock()
        mock_processor.decode.return_value = "a plate of tomatoes"
        mock_processor.return_value = mock_inputs
        mock_model = MagicMock()
        mock_model.generate.return_value = [MagicMock()]
        mock_torch = MagicMock()

        with patch.object(cap_mod, "_BLIP_AVAILABLE", True), \
             patch.object(cap_mod, "_PIL_AVAILABLE", True), \
             patch.object(cap_mod, "_TORCH_AVAILABLE", True), \
             patch.object(cap_mod, "_torch", mock_torch), \
             patch.object(cap_mod, "load_blip", return_value=(mock_processor, mock_model)), \
             patch.object(cap_mod, "select_device", return_value="cpu"):
            result = cap_mod.generate_caption(str(synthetic_image))

        assert "caption" in result
        assert "image" in result
        assert "model" in result
        assert "device" in result

    def test_caption_value(self, synthetic_image: Path):
        import vision.caption as cap_mod
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor = MagicMock()
        mock_processor.decode.return_value = "a fresh salad"
        mock_processor.return_value = mock_inputs
        mock_model = MagicMock()
        mock_model.generate.return_value = [MagicMock()]
        mock_torch = MagicMock()

        with patch.object(cap_mod, "_BLIP_AVAILABLE", True), \
             patch.object(cap_mod, "_PIL_AVAILABLE", True), \
             patch.object(cap_mod, "_TORCH_AVAILABLE", True), \
             patch.object(cap_mod, "_torch", mock_torch), \
             patch.object(cap_mod, "load_blip", return_value=(mock_processor, mock_model)), \
             patch.object(cap_mod, "select_device", return_value="cpu"):
            result = cap_mod.generate_caption(str(synthetic_image))

        assert result["caption"] == "a fresh salad"


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgCaption:
    def test_requires_image(self):
        from vision.caption import parse_args
        with pytest.raises(SystemExit):
            parse_args([])

    def test_default_model(self):
        from vision.caption import parse_args, DEFAULT_MODEL
        args = parse_args(["--image", "img.jpg"])
        assert args.model == DEFAULT_MODEL

    def test_custom_max_tokens(self):
        from vision.caption import parse_args
        args = parse_args(["--image", "img.jpg", "--max-tokens", "100"])
        assert args.max_tokens == 100
