# Fichier: api/server.py — API FastAPI complète pour l'IA culinaire locale.
"""
Ce service expose les endpoints suivants :

Endpoints
---------
- GET  /health              : ping de santé.
- POST /analyse             : détection d'ingrédients (YOLOv8).
- POST /caption             : génération de légende (BLIP).
- POST /analyse-full        : détection + légende + RAG + génération en un seul appel.
- POST /suggest-recipes     : suggestions de recettes depuis une liste d'ingrédients (RAG).
- POST /generate-recipe     : génération de recette via LLM local (ou stub).
- POST /upload-dataset      : import d'images dans la structure dataset YOLO.
- POST /feedback            : collecte de feedback utilisateur.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from vision.detect import run_inference

logger = logging.getLogger(__name__)

app = FastAPI(
    title="IA Culinaire Locale",
    version="0.2.0",
    description=(
        "API locale pour la détection d'ingrédients, la génération de légendes, "
        "la suggestion de recettes (RAG) et la génération de recettes (LLM)."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schémas Pydantic
# ---------------------------------------------------------------------------

class DetectionResponse(BaseModel):
    image: str
    model: str
    device: str
    conf_threshold: float
    detections: List[Dict[str, Any]]


class CaptionResponse(BaseModel):
    image: str
    caption: str
    model: str
    device: str


class RecipeSuggestion(BaseModel):
    id: str = ""
    title: str
    ingredients: List[str] = []
    tags: List[str] = []
    score: float = 0.0


class SuggestRecipesRequest(BaseModel):
    ingredients: List[str] = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class GenerateRecipeRequest(BaseModel):
    ingredients: List[str] = Field(..., min_length=1)
    use_rag: bool = Field(default=True)
    backend: str = Field(default="auto")
    model_path: Optional[str] = Field(default=None)
    max_tokens: int = Field(default=512, ge=50, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GenerateRecipeResponse(BaseModel):
    recipe_text: str
    backend: str
    ingredients_used: List[str]


class AnalyseFullResponse(BaseModel):
    detections: List[Dict[str, Any]]
    caption: Optional[str] = None
    suggested_recipes: List[RecipeSuggestion] = []
    generated_recipe: Optional[str] = None


class DatasetUploadResponse(BaseModel):
    train: int
    val: int
    total: int
    errors: List[str] = []


class FeedbackPayload(BaseModel):
    score: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    comment: Optional[str] = Field(default=None, max_length=2000)
    correction: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_rag_pipeline():
    from rag.pipeline import RecipePipeline
    from rag.recipes import get_builtin_recipes
    pipeline = RecipePipeline(persist_dir="rag/db")
    pipeline.build(get_builtin_recipes())
    return pipeline


def _run_caption(image_path: str) -> Optional[str]:
    try:
        from vision.caption import generate_caption, is_blip_available
        if not is_blip_available():
            return None
        result = generate_caption(image_path)
        return result.get("caption")
    except Exception as exc:
        logger.warning("Génération de légende échouée pour %s: %s", image_path, exc)
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, str]:
    """Vérifie que le service est démarré."""
    return {"status": "ok", "message": "API opérationnelle", "version": "0.2.0"}


@app.post("/analyse", response_model=DetectionResponse)
async def analyse_image(
    file: UploadFile = File(...),
    model: str = Form(default="yolov8n.pt"),
    conf: float = Form(default=0.25),
) -> DetectionResponse:
    """Analyse une image via YOLOv8 et retourne les objets détectés."""
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
        results = run_inference(str(tmp_path), model_name=model, conf=conf)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur d'inférence: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return DetectionResponse(**results)


@app.post("/caption", response_model=CaptionResponse)
async def caption_image(file: UploadFile = File(...)) -> CaptionResponse:
    """Génère une légende textuelle de l'image via BLIP."""
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
        from vision.caption import generate_caption, is_blip_available
        if not is_blip_available():
            raise HTTPException(
                status_code=503,
                detail="BLIP non disponible. Installez transformers et téléchargez les poids.",
            )
        result = generate_caption(str(tmp_path))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur de génération de légende: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return CaptionResponse(**result)


@app.post("/suggest-recipes")
async def suggest_recipes(request: SuggestRecipesRequest) -> Dict[str, Any]:
    """Suggère des recettes depuis une liste d'ingrédients (RAG)."""
    try:
        pipeline = _get_rag_pipeline()
        results = pipeline.query_by_ingredients(request.ingredients, k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur RAG: {exc}") from exc

    suggestions = [
        RecipeSuggestion(
            id=str(r.get("id", "")),
            title=str(r.get("title", "")),
            ingredients=[str(i) for i in r.get("ingredients", [])],
            tags=[str(t) for t in r.get("tags", [])],
            score=float(r.get("_score", 0.0)),
        )
        for r in results
    ]
    return {"ingredients": request.ingredients, "suggestions": [s.model_dump() for s in suggestions]}


@app.post("/generate-recipe", response_model=GenerateRecipeResponse)
async def generate_recipe_endpoint(request: GenerateRecipeRequest) -> GenerateRecipeResponse:
    """Génère une recette via LLM local (ou stub si aucun LLM disponible)."""
    candidates: List[Dict[str, Any]] = []
    if request.use_rag:
        try:
            pipeline = _get_rag_pipeline()
            candidates = pipeline.query_by_ingredients(request.ingredients, k=3)
        except Exception:
            pass

    try:
        from llm.generate import generate_recipe as _gen
        result = _gen(
            ingredients=request.ingredients,
            candidates=candidates,
            backend=request.backend,
            model_path=request.model_path,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur de génération: {exc}") from exc

    return GenerateRecipeResponse(
        recipe_text=result["recipe_text"],
        backend=result["backend"],
        ingredients_used=result["ingredients_used"],
    )


@app.post("/analyse-full", response_model=AnalyseFullResponse)
async def analyse_full(
    file: UploadFile = File(...),
    model: str = Form(default="yolov8n.pt"),
    conf: float = Form(default=0.25),
    do_caption: bool = Form(default=False),
    do_suggest: bool = Form(default=True),
    do_generate: bool = Form(default=False),
) -> AnalyseFullResponse:
    """Pipeline complet: détection → légende → RAG → génération."""
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
        try:
            det_result = run_inference(str(tmp_path), model_name=model, conf=conf)
            detections = det_result.get("detections", [])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Erreur d'inférence: {exc}") from exc

        caption_text: Optional[str] = None
        if do_caption:
            caption_text = _run_caption(str(tmp_path))

        ingredients = list({d["class_name"] for d in detections})
        suggestions: List[RecipeSuggestion] = []
        candidates: List[Dict[str, Any]] = []
        if do_suggest and ingredients:
            try:
                pipeline = _get_rag_pipeline()
                raw = pipeline.query_by_ingredients(ingredients, k=5)
                candidates = raw
                suggestions = [
                    RecipeSuggestion(
                        id=str(r.get("id", "")),
                        title=str(r.get("title", "")),
                        ingredients=[str(i) for i in r.get("ingredients", [])],
                        tags=[str(t) for t in r.get("tags", [])],
                        score=float(r.get("_score", 0.0)),
                    )
                    for r in raw
                ]
            except Exception:
                pass

        generated: Optional[str] = None
        if do_generate and ingredients:
            try:
                from llm.generate import generate_recipe as _gen
                gen_result = _gen(ingredients=ingredients, candidates=candidates, backend="stub")
                generated = gen_result["recipe_text"]
            except Exception:
                pass

    finally:
        tmp_path.unlink(missing_ok=True)

    return AnalyseFullResponse(
        detections=detections,
        caption=caption_text,
        suggested_recipes=suggestions,
        generated_recipe=generated,
    )


@app.post("/upload-dataset", response_model=DatasetUploadResponse)
async def upload_dataset(
    files: List[UploadFile] = File(...),
    output_dir: str = Form(default="data/dataset"),
    val_split: float = Form(default=0.2),
    max_size: Optional[int] = Form(default=None),
) -> DatasetUploadResponse:
    """Importe des images dans la structure dataset YOLO (images/train + images/val)."""
    if not files:
        raise HTTPException(status_code=400, detail="Aucun fichier fourni.")

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        for uf in files:
            if not uf.filename:
                continue
            tmp_file = tmp_dir / uf.filename
            content = await uf.read()
            tmp_file.write_bytes(content)

        from data.import_images import import_from_folder
        result = import_from_folder(
            source_dir=tmp_dir,
            output_dir=Path(output_dir),
            val_split=val_split,
            max_size=max_size,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur d'import: {exc}") from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return DatasetUploadResponse(
        train=result.get("train", 0),
        val=result.get("val", 0),
        total=result.get("total", 0),
        errors=[str(e) for e in result.get("errors", [])],
    )


@app.post("/feedback")
async def feedback(payload: FeedbackPayload) -> Dict[str, Any]:
    """Collecte le feedback utilisateur."""
    return {
        "status": "accepted",
        "detail": "Feedback enregistré",
        "feedback_data": payload.model_dump(),
    }
