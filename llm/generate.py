# Fichier: llm/generate.py — Génération de recettes via un LLM local (ou stub).
"""
Ce module orchestre la génération de recettes textuelles à partir :
- d'ingrédients détectés (par vision/detect.py)
- de recettes candidates récupérées (par rag/pipeline.py)

Il est conçu pour s'intégrer avec plusieurs backends LLM :
1. **llama-cpp-python** : modèles GGUF locaux (recommandé, ex: Mistral, Llama)
2. **gpt4all** : modèles GPT4All locaux
3. **Stub** : génère une recette formatée sans LLM (mode démo / test)

Le backend est sélectionné automatiquement selon les bibliothèques disponibles,
ou peut être forcé via le paramètre `backend`.

Fonctions principales
---------------------
- build_prompt(ingredients, candidates) -> str : construit le prompt structuré.
- generate_recipe(ingredients, candidates, backend, model_path) -> dict : génère la recette.
- list_available_backends() -> list[str] : liste les backends utilisables.

Exemple CLI
-----------
>>> python llm/generate.py --ingredients "tomate, carotte, oignon"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Détection des backends LLM optionnels
# ---------------------------------------------------------------------------

try:
    from llama_cpp import Llama
    _LLAMA_CPP_AVAILABLE = True
except Exception:
    _LLAMA_CPP_AVAILABLE = False

try:
    from gpt4all import GPT4All
    _GPT4ALL_AVAILABLE = True
except Exception:
    _GPT4ALL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Templates de prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
Tu es un chef cuisinier expert. Tu dois proposer une recette détaillée et savoureuse
en utilisant principalement les ingrédients fournis. La recette doit être réalisable
à la maison avec des équipements standard.
"""

_USER_TEMPLATE = """\
Ingrédients disponibles : {ingredients}

Recettes de référence (pour inspiration) :
{candidates_text}

Propose une recette originale et détaillée avec :
1. Titre de la recette
2. Liste complète des ingrédients avec quantités
3. Instructions étape par étape
4. Conseils de chef et variantes possibles
"""


def build_prompt(
    ingredients: List[str],
    candidates: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Construit le prompt structuré pour le LLM.

    Args:
        ingredients: liste d'ingrédients détectés.
        candidates: recettes candidates du RAG (optionnel).

    Returns:
        str: prompt complet prêt à envoyer au LLM.
    """
    ings_text = ", ".join(ingredients) if ingredients else "ingrédients divers"
    if candidates:
        cands = []
        for i, r in enumerate(candidates[:3], 1):
            title = r.get("title", "")
            r_ings = ", ".join(str(x) for x in r.get("ingredients", []))
            cands.append(f"  {i}. {title} ({r_ings})")
        candidates_text = "\n".join(cands)
    else:
        candidates_text = "  (aucune recette de référence)"
    user_msg = _USER_TEMPLATE.format(
        ingredients=ings_text,
        candidates_text=candidates_text,
    )
    return f"[SYSTEM]\n{_SYSTEM_PROMPT}\n[USER]\n{user_msg}\n[ASSISTANT]\n"


def _stub_generate(
    ingredients: List[str],
    candidates: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Génère une recette de substitution structurée sans LLM (mode démo/test).

    Args:
        ingredients: liste d'ingrédients.
        candidates: recettes candidates (pour le titre).

    Returns:
        Texte de recette formaté.
    """
    title = candidates[0]["title"] if candidates else "Recette maison"
    ings = ingredients or ["ingrédients au choix"]
    ings_list = "\n".join(f"  - {i}" for i in ings)
    return (
        f"## {title}\n\n"
        f"### Ingrédients\n{ings_list}\n\n"
        f"### Instructions\n"
        f"  1. Préparez tous vos ingrédients.\n"
        f"  2. Faites revenir les légumes dans un peu d'huile d'olive.\n"
        f"  3. Assaisonnez selon votre goût.\n"
        f"  4. Servez chaud accompagné de votre garniture préférée.\n\n"
        f"### Conseils\n"
        f"  - Ajoutez des herbes fraîches pour rehausser les saveurs.\n"
        f"  - Ce plat se conserve 2 jours au réfrigérateur.\n"
        f"\n_[Recette générée en mode démo — installez llama-cpp-python pour la génération IA]_"
    )


def list_available_backends() -> List[str]:
    """
    Retourne la liste des backends LLM disponibles dans l'environnement courant.

    Returns:
        Liste de chaînes parmi : 'llama_cpp', 'gpt4all', 'stub'.
    """
    backends = ["stub"]  # toujours disponible
    if _LLAMA_CPP_AVAILABLE:
        backends.insert(0, "llama_cpp")
    if _GPT4ALL_AVAILABLE:
        backends.insert(0, "gpt4all")
    return backends


def generate_recipe(
    ingredients: List[str],
    candidates: Optional[List[Dict[str, Any]]] = None,
    backend: str = "auto",
    model_path: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Génère une recette à partir d'ingrédients et de recettes candidates.

    Args:
        ingredients: ingrédients détectés.
        candidates: recettes RAG (contexte pour le LLM).
        backend: 'auto' (choix automatique), 'llama_cpp', 'gpt4all', ou 'stub'.
        model_path: chemin vers le modèle GGUF/GPT4All (requis pour llama_cpp/gpt4all).
        max_tokens: nombre maximum de tokens à générer.
        temperature: créativité du LLM (0 = déterministe, 1 = créatif).

    Returns:
        dict avec:
            - recipe_text: texte de la recette
            - backend: backend utilisé
            - ingredients_used: ingrédients fournis
            - prompt: prompt construit

    Raises:
        ValueError: si le backend demandé n'est pas disponible.
    """
    prompt = build_prompt(ingredients, candidates)

    # Sélection automatique du backend
    if backend == "auto":
        if _LLAMA_CPP_AVAILABLE and model_path:
            backend = "llama_cpp"
        elif _GPT4ALL_AVAILABLE and model_path:
            backend = "gpt4all"
        else:
            backend = "stub"

    recipe_text = ""

    if backend == "llama_cpp":
        if not _LLAMA_CPP_AVAILABLE:
            raise ValueError("llama-cpp-python n'est pas installé. Utilisez : pip install llama-cpp-python")
        if not model_path:
            raise ValueError("model_path est requis pour le backend llama_cpp.")
        llm = Llama(model_path=str(model_path), verbose=False)
        response = llm(prompt, max_tokens=max_tokens, temperature=temperature, echo=False)
        recipe_text = response["choices"][0]["text"].strip()

    elif backend == "gpt4all":
        if not _GPT4ALL_AVAILABLE:
            raise ValueError("gpt4all n'est pas installé. Utilisez : pip install gpt4all")
        if not model_path:
            raise ValueError("model_path est requis pour le backend gpt4all.")
        model = GPT4All(str(model_path))
        with model.chat_session():
            recipe_text = model.generate(prompt, max_tokens=max_tokens, temp=temperature).strip()

    elif backend == "stub":
        recipe_text = _stub_generate(ingredients, candidates)

    else:
        raise ValueError(f"Backend inconnu: '{backend}'. Choisir parmi: {list_available_backends()}")

    return {
        "recipe_text": recipe_text,
        "backend": backend,
        "ingredients_used": ingredients,
        "prompt": prompt,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> Any:
    import argparse
    parser = argparse.ArgumentParser(description="Génération de recettes via LLM local")
    parser.add_argument(
        "--ingredients",
        required=True,
        help="Ingrédients séparés par des virgules (ex: 'tomate, oignon, ail').",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "llama_cpp", "gpt4all", "stub"],
        help="Backend LLM à utiliser.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Chemin vers le fichier modèle GGUF ou GPT4All.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Nombre maximum de tokens à générer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Température de génération (0=déterministe, 1=créatif).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Nombre de recettes RAG à utiliser comme contexte.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ingredients = [i.strip() for i in args.ingredients.split(",")]

    # Tenter de récupérer des recettes du RAG
    candidates: List[Dict[str, Any]] = []
    try:
        from rag.recipes import get_builtin_recipes
        from rag.pipeline import RecipePipeline
        pipeline = RecipePipeline()
        pipeline.build(get_builtin_recipes())
        candidates = pipeline.query_by_ingredients(ingredients, k=args.top)
    except Exception:
        pass  # RAG optionnel

    result = generate_recipe(
        ingredients=ingredients,
        candidates=candidates,
        backend=args.backend,
        model_path=args.model_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(f"\n{'═'*60}")
    print(f"Backend : {result['backend']}")
    print(f"Ingrédients : {', '.join(result['ingredients_used'])}")
    print(f"{'═'*60}\n")
    print(result["recipe_text"])


if __name__ == "__main__":
    main(sys.argv[1:])
