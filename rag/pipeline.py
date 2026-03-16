# Fichier: rag/pipeline.py — Pipeline RAG (ChromaDB + embeddings) pour la recherche de recettes.
"""
Ce module implémente un pipeline RAG (Retrieval-Augmented Generation) pour
retrouver les recettes les plus pertinentes à partir d'une liste d'ingrédients
ou d'une description textuelle.

Architecture
------------
1. Embeddings : sentence-transformers (all-MiniLM-L6-v2 par défaut) ou fallback TF-IDF.
2. Vectorstore : ChromaDB (persisté dans un dossier local).
3. Recherche : top-k recettes par similarité cosinus.

Fonctions principales
---------------------
- RecipePipeline.build(recipes)    : indexe les recettes dans ChromaDB.
- RecipePipeline.query(text, k)    : retourne les k recettes les plus proches.
- RecipePipeline.query_by_ingredients(ingredients, k) : raccourci pratique.

Dégradation gracieuse
---------------------
Si ChromaDB ou sentence-transformers ne sont pas disponibles, le module
retombe automatiquement sur un moteur de recherche TF-IDF (scikit-learn),
ce qui garantit la testabilité sans dépendances lourdes.

Exemple CLI
-----------
>>> python rag/pipeline.py --ingredients "tomate carotte" --top 3
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Détection des dépendances optionnelles
# ---------------------------------------------------------------------------

try:
    import chromadb
    _CHROMA_AVAILABLE = True
except Exception:
    _CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as _np
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "recipes"


# ---------------------------------------------------------------------------
# Moteur de fallback TF-IDF (sans dépendances lourdes)
# ---------------------------------------------------------------------------

class _TfidfEngine:
    """Moteur de recherche TF-IDF — utilisé si ChromaDB / sentence-transformers absents."""

    def __init__(self) -> None:
        self._vectorizer: Optional[Any] = None
        self._matrix: Optional[Any] = None
        self._docs: List[Dict[str, Any]] = []

    def build(self, docs: List[Dict[str, Any]]) -> None:
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn est requis pour le fallback TF-IDF.")
        self._docs = docs
        texts = [str(d.get("text", "")) for d in docs]
        self._vectorizer = TfidfVectorizer()
        self._matrix = self._vectorizer.fit_transform(texts)

    def query(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        if self._vectorizer is None or self._matrix is None:
            raise RuntimeError("Le moteur TF-IDF n'est pas initialisé. Appelez build() d'abord.")
        q_vec = self._vectorizer.transform([text])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = _np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_idx:
            doc = dict(self._docs[idx])
            doc["_score"] = float(scores[idx])
            results.append(doc)
        return results


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

class RecipePipeline:
    """
    Pipeline RAG pour la recherche de recettes culinaires.

    Utilise ChromaDB + sentence-transformers si disponibles,
    sinon retombe sur TF-IDF (sklearn).

    Attributes:
        persist_dir: répertoire de persistance ChromaDB.
        model_id: modèle sentence-transformers.
        collection_name: nom de la collection ChromaDB.
    """

    def __init__(
        self,
        persist_dir: str | Path = "rag/db",
        model_id: str = DEFAULT_MODEL,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.model_id = model_id
        self.collection_name = collection_name
        self._use_chroma = _CHROMA_AVAILABLE and _ST_AVAILABLE
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._embed_model: Optional[Any] = None
        self._fallback = _TfidfEngine()
        self._built = False

    # ------------------------------------------------------------------
    # Internes
    # ------------------------------------------------------------------

    def _get_embed_model(self) -> Any:
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(self.model_id)
        return self._embed_model

    def _embed(self, texts: List[str]) -> List[List[float]]:
        model = self._get_embed_model()
        return model.encode(texts, convert_to_numpy=True).tolist()

    def _get_collection(self) -> Any:
        if self._client is None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(self.collection_name)
        return self._collection

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def build(self, recipes: List[Dict[str, Any]]) -> "RecipePipeline":
        """
        Indexe une liste de recettes dans le vectorstore.

        Args:
            recipes: liste de RecipeDoc (doit avoir les clés id, text).

        Returns:
            self (pour le chaînage).
        """
        if self._use_chroma:
            collection = self._get_collection()
            texts = [str(r.get("text", "")) for r in recipes]
            ids = [str(r.get("id", f"r{i}")) for i, r in enumerate(recipes)]
            embeddings = self._embed(texts)
            metadatas = [
                {
                    "title": str(r.get("title", "")),
                    "tags": json.dumps(r.get("tags", []), ensure_ascii=False),
                    "ingredients": json.dumps(r.get("ingredients", []), ensure_ascii=False),
                }
                for r in recipes
            ]
            # Upsert par lots pour éviter les doublons
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            logger.info("ChromaDB: %d recettes indexées.", len(recipes))
        else:
            self._fallback.build(recipes)
            logger.info("TF-IDF fallback: %d recettes indexées.", len(recipes))
        self._built = True
        return self

    def query(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retourne les k recettes les plus pertinentes pour un texte de requête.

        Args:
            text: texte libre (ingrédients séparés par virgules, description, etc.).
            k: nombre de résultats à retourner.

        Returns:
            Liste de dicts avec les clés de la recette + _score.

        Raises:
            RuntimeError: si build() n'a pas été appelé.
        """
        if not self._built:
            raise RuntimeError("Le pipeline n'est pas initialisé. Appelez build() d'abord.")
        if self._use_chroma:
            collection = self._get_collection()
            embedding = self._embed([text])[0]
            results = collection.query(
                query_embeddings=[embedding],
                n_results=min(k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            docs = []
            for i, doc_text in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i]
                docs.append(
                    {
                        "id": results["ids"][0][i],
                        "title": meta.get("title", ""),
                        "ingredients": json.loads(meta.get("ingredients", "[]")),
                        "tags": json.loads(meta.get("tags", "[]")),
                        "text": doc_text,
                        "_score": float(1.0 - dist),
                    }
                )
            return docs
        else:
            return self._fallback.query(text, k=k)

    def query_by_ingredients(
        self,
        ingredients: List[str],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Recherche des recettes à partir d'une liste d'ingrédients détectés.

        Args:
            ingredients: liste de noms d'ingrédients.
            k: nombre de recettes à retourner.

        Returns:
            Liste de recettes pertinentes avec score.
        """
        query_text = "recette avec: " + ", ".join(ingredients)
        return self.query(query_text, k=k)

    def reset(self) -> None:
        """Supprime la collection ChromaDB (utile pour les tests)."""
        if self._use_chroma and self._client is not None:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = None
        self._fallback = _TfidfEngine()
        self._built = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> Any:
    import argparse
    parser = argparse.ArgumentParser(description="Recherche de recettes par RAG")
    parser.add_argument(
        "--ingredients",
        required=True,
        help="Liste d'ingrédients séparés par des virgules (ex: 'tomate, carotte').",
    )
    parser.add_argument("--top", type=int, default=3, help="Nombre de recettes à afficher.")
    parser.add_argument(
        "--db-dir",
        default="rag/db",
        help="Répertoire de persistance ChromaDB.",
    )
    parser.add_argument(
        "--recipes-json",
        default=None,
        help="Fichier JSON de recettes à indexer (optionnel, utilise les recettes embarquées par défaut).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from rag.recipes import get_builtin_recipes, load_recipes_from_json
    args = parse_args(argv)
    recipes = load_recipes_from_json(args.recipes_json) if args.recipes_json else get_builtin_recipes()
    pipeline = RecipePipeline(persist_dir=args.db_dir)
    pipeline.build(recipes)
    ingredients = [i.strip() for i in args.ingredients.split(",")]
    results = pipeline.query_by_ingredients(ingredients, k=args.top)
    for i, r in enumerate(results, 1):
        print(f"\n{'─'*50}")
        print(f"#{i} {r.get('title', '')}  (score: {r.get('_score', 0):.3f})")
        if r.get("ingredients"):
            print(f"  Ingrédients: {', '.join(str(x) for x in r['ingredients'])}")
        if r.get("tags"):
            print(f"  Tags: {', '.join(str(t) for t in r['tags'])}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
