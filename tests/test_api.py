# Fichier: tests/test_api.py — Tests d'intégration pour api/server.py.
"""
Vérifie les endpoints FastAPI (health, analyse, feedback) à l'aide du TestClient
Starlette. La fonction run_inference est systématiquement mockée pour éviter
toute dépendance envers ultralytics/torch.
"""

from __future__ import annotations

import io
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
    """Retourne un TestClient FastAPI avec run_inference mocké."""
    from api.server import app
    return TestClient(app)


@pytest.fixture()
def jpeg_bytes() -> bytes:
    """Génère des bytes d'une image JPEG 32×32 synthétique."""
    buf = io.BytesIO()
    img = Image.new("RGB", (32, 32), color=(180, 60, 60))
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient):
        """GET /health renvoie 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_field(self, client: TestClient):
        """La réponse JSON doit contenir status='ok'."""
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_message_field_present(self, client: TestClient):
        """La réponse doit contenir un champ 'message'."""
        data = client.get("/health").json()
        assert "message" in data


# ---------------------------------------------------------------------------
# POST /analyse
# ---------------------------------------------------------------------------

class TestAnalyseEndpoint:
    def test_no_file_returns_422(self, client: TestClient):
        """Envoyer une requête sans fichier retourne 422 (validation FastAPI)."""
        response = client.post("/analyse")
        assert response.status_code == 422

    def test_valid_image_returns_200(self, client: TestClient, jpeg_bytes: bytes):
        """Une image JPEG valide retourne 200 avec les données mockées."""
        with patch("api.server.run_inference", return_value=FAKE_INFERENCE_RESULT):
            response = client.post(
                "/analyse",
                files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")},
            )
        assert response.status_code == 200

    def test_response_structure(self, client: TestClient, jpeg_bytes: bytes):
        """La réponse doit avoir toutes les clés du modèle DetectionResponse."""
        with patch("api.server.run_inference", return_value=FAKE_INFERENCE_RESULT):
            data = client.post(
                "/analyse",
                files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")},
            ).json()
        for key in ("image", "model", "device", "conf_threshold", "detections"):
            assert key in data

    def test_detections_are_list(self, client: TestClient, jpeg_bytes: bytes):
        """Le champ 'detections' doit être une liste."""
        with patch("api.server.run_inference", return_value=FAKE_INFERENCE_RESULT):
            data = client.post(
                "/analyse",
                files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")},
            ).json()
        assert isinstance(data["detections"], list)

    def test_inference_error_returns_500(self, client: TestClient, jpeg_bytes: bytes):
        """Si run_inference lève une exception, l'API retourne 500."""
        with patch("api.server.run_inference", side_effect=RuntimeError("boom")):
            response = client.post(
                "/analyse",
                files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")},
            )
        assert response.status_code == 500

    def test_detection_class_names(self, client: TestClient, jpeg_bytes: bytes):
        """Les noms de classes doivent être préservés dans la réponse."""
        with patch("api.server.run_inference", return_value=FAKE_INFERENCE_RESULT):
            data = client.post(
                "/analyse",
                files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")},
            ).json()
        class_names = [d["class_name"] for d in data["detections"]]
        assert "tomate" in class_names

    def test_empty_detections_response(self, client: TestClient, jpeg_bytes: bytes):
        """L'API gère correctement le cas sans détections."""
        empty_result = {**FAKE_INFERENCE_RESULT, "detections": []}
        with patch("api.server.run_inference", return_value=empty_result):
            data = client.post(
                "/analyse",
                files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")},
            ).json()
        assert data["detections"] == []


# ---------------------------------------------------------------------------
# POST /feedback
# ---------------------------------------------------------------------------

class TestFeedbackEndpoint:
    def test_accepted_with_score(self, client: TestClient):
        """Un feedback avec score retourne 200 et status='accepted'."""
        response = client.post("/feedback", json={"score": 4.5})
        assert response.status_code == 200
        assert response.json()["status"] == "accepted"

    def test_accepted_with_comment(self, client: TestClient):
        """Un feedback avec commentaire est accepté."""
        response = client.post("/feedback", json={"comment": "Très bien détecté !"})
        assert response.status_code == 200

    def test_accepted_empty_payload(self, client: TestClient):
        """Un payload vide est valide (tous les champs sont optionnels)."""
        response = client.post("/feedback", json={})
        assert response.status_code == 200

    def test_score_out_of_range_rejected(self, client: TestClient):
        """Un score supérieur à 5 doit être rejeté (422)."""
        response = client.post("/feedback", json={"score": 10.0})
        assert response.status_code == 422

    def test_negative_score_rejected(self, client: TestClient):
        """Un score négatif doit être rejeté (422)."""
        response = client.post("/feedback", json={"score": -1.0})
        assert response.status_code == 422

    def test_feedback_data_in_response(self, client: TestClient):
        """La réponse doit contenir les données du feedback soumis."""
        payload = {"score": 3.0, "comment": "Correct"}
        data = client.post("/feedback", json=payload).json()
        assert "feedback_data" in data
        assert data["feedback_data"]["score"] == pytest.approx(3.0)

    def test_comment_too_long_rejected(self, client: TestClient):
        """Un commentaire dépassant 2000 caractères doit être rejeté (422)."""
        response = client.post("/feedback", json={"comment": "x" * 2001})
        assert response.status_code == 422
