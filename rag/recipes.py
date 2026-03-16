# Fichier: rag/recipes.py — Données de recettes et utilitaires de chargement.
"""
Ce module fournit :
- Un jeu de données de recettes culinaires embarqué (20 recettes types).
- Des fonctions pour charger des recettes depuis un fichier JSON externe.
- Un formatage normalisé (RecipeDoc) prêt à être ingéré par le pipeline RAG.

Structure d'une recette (RecipeDoc)
------------------------------------
{
    "id": "r001",
    "title": "Ratatouille Provençale",
    "ingredients": ["tomate", "courgette", "aubergine", "poivron", "oignon"],
    "instructions": "...",
    "tags": ["végétarien", "méditerranéen"],
    "text": "<titre> + <ingrédients> + <instructions>" (champ composite pour l'embedding)
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

RecipeDoc = Dict[str, object]


# ---------------------------------------------------------------------------
# Données embarquées — 20 recettes représentatives
# ---------------------------------------------------------------------------

_BUILTIN_RECIPES: List[RecipeDoc] = [
    {
        "id": "r001",
        "title": "Ratatouille Provençale",
        "ingredients": ["tomate", "courgette", "aubergine", "poivron rouge", "poivron vert", "oignon", "ail", "huile d'olive", "thym", "basilic"],
        "instructions": "Couper tous les légumes en rondelles. Faire revenir l'oignon et l'ail dans l'huile d'olive. Ajouter les légumes dans l'ordre de cuisson. Assaisonner avec thym et basilic. Mijoter 40 minutes à feu doux.",
        "tags": ["végétarien", "méditerranéen", "été"],
    },
    {
        "id": "r002",
        "title": "Soupe aux tomates et basilic",
        "ingredients": ["tomate", "oignon", "ail", "basilic frais", "bouillon de légumes", "crème fraîche", "huile d'olive"],
        "instructions": "Faire suer l'oignon et l'ail. Ajouter les tomates coupées et le bouillon. Cuire 20 minutes. Mixer et ajouter le basilic. Finir avec la crème fraîche.",
        "tags": ["végétarien", "soupe", "tomate"],
    },
    {
        "id": "r003",
        "title": "Gratin de courgettes au fromage",
        "ingredients": ["courgette", "fromage râpé", "crème fraîche", "ail", "noix de muscade", "sel", "poivre"],
        "instructions": "Couper les courgettes en rondelles. Disposer dans un plat beurré. Mélanger crème, fromage, ail et muscade. Verser sur les courgettes. Gratiner 25 minutes à 180°C.",
        "tags": ["végétarien", "gratin", "courgette"],
    },
    {
        "id": "r004",
        "title": "Salade de carottes râpées",
        "ingredients": ["carotte", "citron", "huile d'olive", "persil", "raisins secs", "sel", "poivre"],
        "instructions": "Râper les carottes. Préparer une vinaigrette citron-huile. Mélanger avec les raisins secs et le persil haché. Assaisonner et réfrigérer 30 minutes.",
        "tags": ["végétarien", "salade", "carotte", "cru"],
    },
    {
        "id": "r005",
        "title": "Velouté de brocoli",
        "ingredients": ["brocoli", "pomme de terre", "oignon", "bouillon de légumes", "crème fraîche", "noix de muscade"],
        "instructions": "Faire revenir l'oignon. Ajouter le brocoli, la pomme de terre et le bouillon. Cuire 20 minutes. Mixer finement. Ajouter la crème et la muscade.",
        "tags": ["végétarien", "soupe", "brocoli"],
    },
    {
        "id": "r006",
        "title": "Omelette aux champignons",
        "ingredients": ["oeuf", "champignon de Paris", "beurre", "persil", "sel", "poivre", "crème fraîche"],
        "instructions": "Faire sauter les champignons au beurre. Battre les œufs avec la crème. Cuire l'omelette dans une poêle beurrée. Garnir avec les champignons et le persil.",
        "tags": ["rapide", "oeuf", "champignon"],
    },
    {
        "id": "r007",
        "title": "Poulet rôti aux herbes",
        "ingredients": ["poulet", "ail", "thym", "romarin", "citron", "huile d'olive", "beurre", "sel", "poivre"],
        "instructions": "Préchauffer le four à 200°C. Mélanger beurre, herbes et ail. Badigeonner le poulet. Ajouter le citron en quartiers. Rôtir 1h15 en arrosant régulièrement.",
        "tags": ["viande", "poulet", "four"],
    },
    {
        "id": "r008",
        "title": "Tarte aux pommes",
        "ingredients": ["pomme", "pâte brisée", "sucre", "cannelle", "beurre", "oeuf", "confiture d'abricot"],
        "instructions": "Étaler la pâte dans un moule. Peler et couper les pommes en lamelles. Disposer en rosace. Saupoudrer de sucre et cannelle. Dorer au beurre. Cuire 35 minutes à 180°C. Lustrer à la confiture.",
        "tags": ["dessert", "tarte", "pomme"],
    },
    {
        "id": "r009",
        "title": "Poêlée de légumes d'été",
        "ingredients": ["courgette", "poivron", "aubergine", "tomate cerise", "oignon", "ail", "huile d'olive", "herbes de Provence"],
        "instructions": "Découper tous les légumes en cubes. Faire chauffer l'huile. Faire revenir l'oignon et l'ail. Ajouter les légumes durs en premier, puis les plus tendres. Assaisonner. Cuire 15 minutes.",
        "tags": ["végétarien", "légumes", "été", "poêlée"],
    },
    {
        "id": "r010",
        "title": "Risotto aux champignons",
        "ingredients": ["riz arborio", "champignon", "oignon", "vin blanc", "bouillon de légumes", "parmesan", "beurre", "persil"],
        "instructions": "Faire revenir l'oignon dans le beurre. Ajouter le riz et nacrer. Déglacer au vin blanc. Ajouter le bouillon chaud louche par louche. À mi-cuisson, ajouter les champignons. Finir avec le parmesan et le beurre.",
        "tags": ["végétarien", "riz", "champignon", "risotto"],
    },
    {
        "id": "r011",
        "title": "Smoothie banane-épinard",
        "ingredients": ["banane", "épinard", "lait de coco", "gingembre", "miel", "citron vert"],
        "instructions": "Mixer tous les ingrédients jusqu'à obtenir une texture lisse. Servir immédiatement bien frais.",
        "tags": ["végétalien", "boisson", "banane", "rapide"],
    },
    {
        "id": "r012",
        "title": "Crêpes sucrées",
        "ingredients": ["farine", "oeuf", "lait", "beurre", "sucre", "sel", "extrait de vanille"],
        "instructions": "Mélanger farine, sucre, sel. Ajouter les œufs puis le lait progressivement. Incorporer le beurre fondu et la vanille. Laisser reposer 30 minutes. Cuire à la poêle.",
        "tags": ["dessert", "crêpe", "classique"],
    },
    {
        "id": "r013",
        "title": "Salade niçoise",
        "ingredients": ["thon", "oeuf dur", "tomate", "haricot vert", "olive noire", "anchois", "poivron", "oignon rouge", "vinaigrette"],
        "instructions": "Cuire les haricots verts al dente. Assembler tous les ingrédients dans un plat. Arroser de vinaigrette.",
        "tags": ["salade", "méditerranéen", "thon"],
    },
    {
        "id": "r014",
        "title": "Curry de légumes",
        "ingredients": ["pomme de terre", "carotte", "pois chiche", "tomate", "lait de coco", "pâte de curry", "oignon", "ail", "gingembre", "coriandre"],
        "instructions": "Faire revenir l'oignon, l'ail et le gingembre. Ajouter la pâte de curry. Incorporer les légumes et le lait de coco. Mijoter 25 minutes. Garnir de coriandre.",
        "tags": ["végétalien", "curry", "épicé", "indien"],
    },
    {
        "id": "r015",
        "title": "Gaspacho andalou",
        "ingredients": ["tomate", "concombre", "poivron rouge", "ail", "pain rassis", "huile d'olive", "vinaigre de xérès", "sel"],
        "instructions": "Tremper le pain dans l'eau. Mixer tous les légumes avec l'huile et le vinaigre. Passer au tamis. Assaisonner. Réfrigérer 2 heures avant de servir.",
        "tags": ["végétarien", "soupe froide", "été", "espagnol"],
    },
    {
        "id": "r016",
        "title": "Tarte aux poireaux et fromage de chèvre",
        "ingredients": ["poireau", "fromage de chèvre", "oeuf", "crème fraîche", "pâte brisée", "noix de muscade", "beurre", "sel", "poivre"],
        "instructions": "Faire fondre les poireaux émincés dans le beurre. Étaler la pâte. Mélanger les poireaux avec les œufs battus et la crème. Parsemer de fromage de chèvre. Cuire 30 minutes à 180°C.",
        "tags": ["végétarien", "tarte", "poireau"],
    },
    {
        "id": "r017",
        "title": "Saumon en papillote aux légumes",
        "ingredients": ["saumon", "courgette", "carotte", "tomate cerise", "citron", "herbes fraîches", "huile d'olive", "sel", "poivre"],
        "instructions": "Disposer le saumon sur une feuille d'aluminium. Ajouter les légumes taillés en julienne, le citron et les herbes. Refermer la papillote. Cuire 20 minutes à 200°C.",
        "tags": ["poisson", "saumon", "papillote", "santé"],
    },
    {
        "id": "r018",
        "title": "Taboulé libanais",
        "ingredients": ["boulgour", "tomate", "concombre", "persil plat", "menthe", "oignon vert", "citron", "huile d'olive"],
        "instructions": "Tremper le boulgour dans l'eau froide 30 minutes. Hacher finement persil et menthe. Couper les légumes en dés. Tout mélanger avec le citron et l'huile. Assaisonner.",
        "tags": ["végétalien", "libanais", "salade", "céréales"],
    },
    {
        "id": "r019",
        "title": "Soufflé au fromage",
        "ingredients": ["fromage râpé", "oeuf", "beurre", "farine", "lait", "noix de muscade", "sel", "poivre"],
        "instructions": "Préparer une béchamel. Y incorporer le fromage et les jaunes. Battre les blancs en neige ferme. Incorporer délicatement. Verser dans des ramequins beurrés et farinés. Cuire 15 minutes à 190°C sans ouvrir le four.",
        "tags": ["végétarien", "fromage", "soufflé", "four"],
    },
    {
        "id": "r020",
        "title": "Pâtes au pesto de basilic",
        "ingredients": ["pâtes", "basilic frais", "pignons de pin", "parmesan", "ail", "huile d'olive", "sel"],
        "instructions": "Mixer basilic, pignons, parmesan, ail et huile jusqu'à obtenir une pâte homogène. Cuire les pâtes al dente. Égoutter en conservant un peu d'eau de cuisson. Mélanger avec le pesto en ajoutant l'eau si nécessaire.",
        "tags": ["végétarien", "pâtes", "pesto", "rapide"],
    },
]


def _make_text(recipe: RecipeDoc) -> str:
    """Construit le champ texte composite utilisé pour l'embedding."""
    title = str(recipe.get("title", ""))
    ingredients = ", ".join(str(i) for i in recipe.get("ingredients", []))
    instructions = str(recipe.get("instructions", ""))
    tags = ", ".join(str(t) for t in recipe.get("tags", []))
    return f"{title}. Ingrédients: {ingredients}. {instructions} Tags: {tags}"


def get_builtin_recipes() -> List[RecipeDoc]:
    """
    Retourne les recettes embarquées avec le champ texte composite.

    Returns:
        Liste de RecipeDoc enrichis du champ 'text'.
    """
    result = []
    for r in _BUILTIN_RECIPES:
        doc = dict(r)
        doc["text"] = _make_text(doc)
        result.append(doc)
    return result


def load_recipes_from_json(path: str | Path) -> List[RecipeDoc]:
    """
    Charge un fichier JSON contenant une liste de recettes.

    Le fichier doit être un tableau JSON dont chaque élément est un objet avec
    au minimum les clés : id, title, ingredients (liste), instructions.

    Args:
        path: chemin du fichier JSON.

    Returns:
        Liste de RecipeDoc enrichis du champ 'text'.

    Raises:
        FileNotFoundError: si le fichier est absent.
        ValueError: si le format JSON est invalide.
    """
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Fichier de recettes introuvable: {path}")
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON invalide dans {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError(f"Le fichier {path} doit être un tableau JSON de recettes.")

    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        doc = dict(item)
        doc.setdefault("id", f"ext_{len(result)}")
        doc.setdefault("ingredients", [])
        doc.setdefault("instructions", "")
        doc.setdefault("tags", [])
        doc["text"] = _make_text(doc)
        result.append(doc)
    return result


def _ingredients_match(query_ing: str, recipe_ing: str) -> bool:
    """
    Vérifie si un ingrédient de requête correspond à un ingrédient de recette.
    Utilise une correspondance partielle bidirectionnelle (ex: 'tomate' ↔ 'tomate cerise').

    Args:
        query_ing: ingrédient recherché (en minuscules).
        recipe_ing: ingrédient de la recette (en minuscules).

    Returns:
        True si l'un contient l'autre.
    """
    return query_ing in recipe_ing or recipe_ing in query_ing


def filter_by_ingredients(
    recipes: List[RecipeDoc],
    ingredients: List[str],
    min_match: int = 1,
) -> List[RecipeDoc]:
    """
    Filtre les recettes qui contiennent au moins `min_match` des ingrédients fournis.

    Args:
        recipes: liste de RecipeDoc.
        ingredients: ingrédients détectés (noms normalisés en minuscules).
        min_match: nombre minimum d'ingrédients correspondants.

    Returns:
        Sous-liste triée par nombre de correspondances décroissant.
    """
    query = {ing.lower().strip() for ing in ingredients}
    scored = []
    for recipe in recipes:
        recipe_ings = {str(i).lower() for i in recipe.get("ingredients", [])}
        matches = sum(1 for q in query if any(_ingredients_match(q, ri) for ri in recipe_ings))
        if matches >= min_match:
            scored.append((matches, recipe))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored]
