# Fichier: tests/test_api_extended.py — Tests d'intégration pour les nouveaux endpoints.
"""
Tests des endpoints /caption, /suggest-recipes, /generate-recipe,
/analyse-full et /upload-dataset via FastAPI TestClient.
"""

from __future__ import annotations

import io
from typing import Any, Dict
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.conftest import FAKE_INFERENCE_RESULT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client() -> TestClient:
    from api.server import app
    return TestClient(app)


@pytest.fixture()
def jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(100, 200, 100)).save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# GET /health (mise à jour version 0.2.0)
# ---------------------------------------------------------------------------

class TestHealthV2:
    def test_version_field(self, client: TestClient):
        data = client.get("/health").json()
        assert "version" in data


# ---------------------------------------------------------------------------
# POST /caption
# ---------------------------------------------------------------------------

class TestCaptionEndpoint:
    def test_no_file_returns_422(self, client: TestClient):
        assert client.post("/caption").status_code == 422

    def test_blip_unavailable_returns_503(self, client: TestClient, jpeg_bytes: bytes):
        with patch("api.server._run_caption", return_value=None), \
             patch("vision.caption.is_blip_available", return_value=False):
            response = client.post(
                "/caption",
                files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")},
            )
        assert response.status_code == 503

    def test_returns_200_when_available(self, client: TestClient, jpeg_bytes: bytes):
        mock_result = {
            "image": "test.jpg",
            "caption": "a photo of vegetables",
            "model": "Salesforce/blip-image-captioning-base",
            "device": "cpu",
        }
        with patch("api.server.generate_caption", mock_result.__class__.get, create=True):
            with patch("vision.caption.is_blip_available", return_value=True), \
                 patch("vision.caption.generate_caption", return_value=mock_result):
                response = client.post(
                    "/caption",
                    files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")},
                )
        assert response.status_code == 200

    def test_caption_response_structure(self, client: TestClient, jpeg_bytes: bytes):
        mock_result = {
            "image": "test.jpg",
            "caption": "a plate of tomatoes",
            "model": "Salesforce/blip-image-captioning-base",
            "device": "cpu",
        }
        with patch("vision.caption.is_blip_available", return_value=True), \
             patch("vision.caption.generate_caption", return_value=mock_result):
            data = client.post(
                "/caption",
                files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")},
            ).json()
        if "caption" in data:
            assert isinstance(data["caption"], str)


# ---------------------------------------------------------------------------
# POST /suggest-recipes
# ---------------------------------------------------------------------------

class TestSuggestRecipesEndpoint:
    def test_returns_200(self, client: TestClient):
        with patch("api.server._get_rag_pipeline") as mock_rag:
            mock_pipeline = mock_rag.return_value
            mock_pipeline.query_by_ingredients.return_value = [
                {
                    "id": "r001",
                    "title": "Ratatouille",
                    "ingredients": ["tomate", "courgette"],
                    "tags": ["végétarien"],
                    "_score": 0.9,
                }
            ]
            response = client.post(
                "/suggest-recipes",
                json={"ingredients": ["tomate", "courgette"], "top_k": 3},
            )
        assert response.status_code == 200

    def test_returns_suggestions_list(self, client: TestClient):
        with patch("api.server._get_rag_pipeline") as mock_rag:
            mock_pipeline = mock_rag.return_value
            mock_pipeline.query_by_ingredients.return_value = [
                {"id": "r001", "title": "Ratatouille", "ingredients": ["tomate"], "tags": [], "_score": 0.8}
            ]
            data = client.post(
                "/suggest-recipes",
                json={"ingredients": ["tomate"]},
            ).json()
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)

    def test_empty_ingredients_rejected(self, client: TestClient):
        response = client.post("/suggest-recipes", json={"ingredients": []})
        assert response.status_code == 422

    def test_suggestions_have_title(self, client: TestClient):
        with patch("api.server._get_rag_pipeline") as mock_rag:
            mock_pipeline = mock_rag.return_value
            mock_pipeline.query_by_ingredients.return_value = [
                {"id": "r001", "title": "Soupe tomate", "ingredients": ["tomate"], "tags": [], "_score": 0.7}
            ]
            data = client.post("/suggest-recipes", json={"ingredients": ["tomate"]}).json()
        assert data["suggestions"][0]["title"] == "Soupe tomate"

    def test_top_k_validated(self, client: TestClient):
        response = client.post("/suggest-recipes", json={"ingredients": ["tomate"], "top_k": 0})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /generate-recipe
# ---------------------------------------------------------------------------

class TestGenerateRecipeEndpoint:
    def _mock_gen_result(self) -> Dict[str, Any]:
        return {
            "recipe_text": "## Ratatouille\n...",
            "backend": "stub",
            "ingredients_used": ["tomate"],
            "prompt": "...",
        }

    def test_returns_200(self, client: TestClient):
        with patch("api.server._get_rag_pipeline") as mock_rag, \
             patch("llm.generate.generate_recipe", return_value=self._mock_gen_result()):
            mock_rag.return_value.query_by_ingredients.return_value = []
            response = client.post(
                "/generate-recipe",
                json={"ingredients": ["tomate"], "backend": "stub"},
            )
        assert response.status_code == 200

    def test_response_has_recipe_text(self, client: TestClient):
        with patch("api.server._get_rag_pipeline") as mock_rag, \
             patch("llm.generate.generate_recipe", return_value=self._mock_gen_result()):
            mock_rag.return_value.query_by_ingredients.return_value = []
            data = client.post(
                "/generate-recipe",
                json={"ingredients": ["tomate"], "backend": "stub"},
            ).json()
        assert "recipe_text" in data

    def test_empty_ingredients_rejected(self, client: TestClient):
        response = client.post("/generate-recipe", json={"ingredients": []})
        assert response.status_code == 422

    def test_invalid_temperature_rejected(self, client: TestClient):
        response = client.post(
            "/generate-recipe",
            json={"ingredients": ["tomate"], "temperature": 5.0},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /analyse-full
# ---------------------------------------------------------------------------

class TestAnalyseFullEndpoint:
    def test_returns_200(self, client: TestClient, jpeg_bytes: bytes):
        with patch("api.server.run_inference", return_value=FAKE_INFERENCE_RESULT), \
             patch("api.server._get_rag_pipeline") as mock_rag:
            mock_rag.return_value.query_by_ingredients.return_value = []
            response = client.post(
                "/analyse-full",
                files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")},
            )
        assert response.status_code == 200

    def test_detections_in_response(self, client: TestClient, jpeg_bytes: bytes):
        with patch("api.server.run_inference", return_value=FAKE_INFERENCE_RESULT), \
             patch("api.server._get_rag_pipeline") as mock_rag:
            mock_rag.return_value.query_by_ingredients.return_value = []
            data = client.post(
                "/analyse-full",
                files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")},
            ).json()
        assert "detections" in data
        assert isinstance(data["detections"], list)

    def test_no_file_returns_422(self, client: TestClient):
        assert client.post("/analyse-full").status_code == 422

    def test_suggested_recipes_present(self, client: TestClient, jpeg_bytes: bytes):
        mock_rag_result = [
            {"id": "r1", "title": "Soupe", "ingredients": ["tomate"], "tags": [], "_score": 0.8}
        ]
        with patch("api.server.run_inference", return_value=FAKE_INFERENCE_RESULT), \
             patch("api.server._get_rag_pipeline") as mock_rag:
            mock_rag.return_value.query_by_ingredients.return_value = mock_rag_result
            data = client.post(
                "/analyse-full",
                files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")},
                data={"do_suggest": "true"},
            ).json()
        assert "suggested_recipes" in data


# ---------------------------------------------------------------------------
# POST /upload-dataset
# ---------------------------------------------------------------------------

class TestUploadDatasetEndpoint:
    def test_no_files_returns_422(self, client: TestClient):
        assert client.post("/upload-dataset").status_code == 422

    def test_returns_200_with_image(self, client: TestClient, jpeg_bytes: bytes, tmp_path):
        with patch("data.import_images.import_from_folder") as mock_import:
            mock_import.return_value = {"train": 1, "val": 0, "total": 1, "errors": [], "files": []}
            response = client.post(
                "/upload-dataset",
                files={"files": ("photo.jpg", jpeg_bytes, "image/jpeg")},
                data={"output_dir": str(tmp_path / "ds"), "val_split": "0.2"},
            )
        assert response.status_code == 200

    def test_response_structure(self, client: TestClient, jpeg_bytes: bytes, tmp_path):
        with patch("data.import_images.import_from_folder") as mock_import:
            mock_import.return_value = {"train": 1, "val": 0, "total": 1, "errors": [], "files": []}
            data = client.post(
                "/upload-dataset",
                files={"files": ("photo.jpg", jpeg_bytes, "image/jpeg")},
                data={"output_dir": str(tmp_path / "ds")},
            ).json()
        assert "train" in data
        assert "val" in data
        assert "total" in data
