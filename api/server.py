# Fichier: api/server.py — API FastAPI minimale pour analyser des images culinaires localement.
"""
Ce service expose un endpoint `/analyse` acceptant une image, exécute la détection
YOLOv8 via `vision/detect.py`, et renvoie un JSON structuré (ingrédients/objets détectés).
L'architecture est pensée pour être étendue ultérieurement avec la caption BLIP,
le RAG sur recettes et un LLM local pour la génération finale.

Endpoints
---------
- GET /health : ping de santé simple.
- POST /analyse : upload d'image (multipart/form-data) -> détections JSON.
- POST /feedback (placeholder) : stockage ultérieur des retours utilisateurs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from vision.detect import run_inference

app = FastAPI(title="IA Culinaire Locale", version="0.1.0")

# CORS permissif pour faciliter les tests frontend locaux (à restreindre en prod).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectionResponse(BaseModel):
    """
    Modèle de réponse standard pour l'endpoint /analyse.
    """

    image: str
    model: str
    device: str
    conf_threshold: float
    detections: List[Dict[str, Any]]


@app.get("/health")
async def health() -> Dict[str, str]:
    """
    Vérifie que le service est démarré.

    Returns:
        dict: message de statut.
    """
    return {"status": "ok", "message": "API opérationnelle"}


@app.post("/analyse", response_model=DetectionResponse)
async def analyse_image(file: UploadFile = File(...)) -> DetectionResponse:
    """
    Analyse une image envoyée en multipart/form-data.

    Args:
        file: image uploadée (JPEG/PNG recommandé).

    Returns:
        DetectionResponse: JSON structuré avec les boîtes détectées.

    Raises:
        HTTPException: si le fichier est invalide ou l'inférence échoue.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Aucun fichier fourni.")

    suffix = Path(file.filename).suffix or ".jpg"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Impossible de sauvegarder le fichier: {exc}") from exc

    try:
        results = run_inference(str(tmp_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur d'inférence: {exc}") from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return DetectionResponse(**results)


@app.post("/feedback")
async def feedback(payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Point d'extension pour recevoir le feedback utilisateur.
    (Non persisté ici, à compléter avec une base locale SQLite/CSV.)

    Args:
        payload: contenu libre envoyé par le frontend.

    Returns:
        dict: message de confirmation.
    """
    return {"status": "accepted", "detail": "Feedback enregistré (placeholder)"}
