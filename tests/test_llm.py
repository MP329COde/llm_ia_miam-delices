# Fichier: tests/test_llm.py — Tests pour llm/generate.py.
"""
Ces tests couvrent la génération de recettes sans LLM réel (mode stub uniquement).
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from tests.conftest import FAKE_DETECTIONS


SAMPLE_INGREDIENTS = ["tomate", "carotte", "oignon"]
SAMPLE_CANDIDATES = [
    {
        "id": "r001",
        "title": "Soupe de légumes",
        "ingredients": ["tomate", "carotte", "oignon"],
        "tags": ["végétarien"],
        "_score": 0.9,
    }
]


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_contains_ingredients(self):
        from llm.generate import build_prompt
        prompt = build_prompt(SAMPLE_INGREDIENTS)
        assert "tomate" in prompt
        assert "carotte" in prompt

    def test_contains_candidates_title(self):
        from llm.generate import build_prompt
        prompt = build_prompt(SAMPLE_INGREDIENTS, SAMPLE_CANDIDATES)
        assert "Soupe de légumes" in prompt

    def test_handles_empty_candidates(self):
        from llm.generate import build_prompt
        prompt = build_prompt(SAMPLE_INGREDIENTS, candidates=None)
        assert "aucune recette de référence" in prompt

    def test_handles_empty_ingredients(self):
        from llm.generate import build_prompt
        prompt = build_prompt([])
        assert isinstance(prompt, str)
        assert len(prompt) > 10

    def test_contains_system_prompt(self):
        from llm.generate import build_prompt, _SYSTEM_PROMPT
        prompt = build_prompt(SAMPLE_INGREDIENTS)
        assert "chef cuisinier" in prompt.lower() or "[SYSTEM]" in prompt


# ---------------------------------------------------------------------------
# list_available_backends
# ---------------------------------------------------------------------------

class TestListAvailableBackends:
    def test_stub_always_present(self):
        from llm.generate import list_available_backends
        assert "stub" in list_available_backends()

    def test_returns_list(self):
        from llm.generate import list_available_backends
        assert isinstance(list_available_backends(), list)


# ---------------------------------------------------------------------------
# generate_recipe (stub backend)
# ---------------------------------------------------------------------------

class TestGenerateRecipeStub:
    def test_stub_returns_dict(self):
        from llm.generate import generate_recipe
        result = generate_recipe(SAMPLE_INGREDIENTS, backend="stub")
        assert isinstance(result, dict)

    def test_stub_has_recipe_text(self):
        from llm.generate import generate_recipe
        result = generate_recipe(SAMPLE_INGREDIENTS, backend="stub")
        assert "recipe_text" in result
        assert len(result["recipe_text"]) > 20

    def test_stub_backend_field(self):
        from llm.generate import generate_recipe
        result = generate_recipe(SAMPLE_INGREDIENTS, backend="stub")
        assert result["backend"] == "stub"

    def test_stub_ingredients_preserved(self):
        from llm.generate import generate_recipe
        result = generate_recipe(SAMPLE_INGREDIENTS, backend="stub")
        assert result["ingredients_used"] == SAMPLE_INGREDIENTS

    def test_stub_with_candidates(self):
        from llm.generate import generate_recipe
        result = generate_recipe(SAMPLE_INGREDIENTS, candidates=SAMPLE_CANDIDATES, backend="stub")
        assert "Soupe de légumes" in result["recipe_text"]

    def test_stub_prompt_in_result(self):
        from llm.generate import generate_recipe
        result = generate_recipe(SAMPLE_INGREDIENTS, backend="stub")
        assert "prompt" in result

    def test_auto_falls_back_to_stub(self):
        """Sans llama_cpp ni gpt4all, 'auto' doit choisir 'stub'."""
        from llm.generate import generate_recipe
        import llm.generate as gen_mod
        with patch.object(gen_mod, "_LLAMA_CPP_AVAILABLE", False), \
             patch.object(gen_mod, "_GPT4ALL_AVAILABLE", False):
            result = generate_recipe(SAMPLE_INGREDIENTS, backend="auto")
            assert result["backend"] == "stub"

    def test_unknown_backend_raises(self):
        from llm.generate import generate_recipe
        with pytest.raises(ValueError):
            generate_recipe(SAMPLE_INGREDIENTS, backend="unknown_backend")

    def test_llama_cpp_unavailable_raises(self):
        from llm.generate import generate_recipe
        import llm.generate as gen_mod
        with patch.object(gen_mod, "_LLAMA_CPP_AVAILABLE", False):
            with pytest.raises(ValueError):
                generate_recipe(SAMPLE_INGREDIENTS, backend="llama_cpp")

    def test_gpt4all_unavailable_raises(self):
        from llm.generate import generate_recipe
        import llm.generate as gen_mod
        with patch.object(gen_mod, "_GPT4ALL_AVAILABLE", False):
            with pytest.raises(ValueError):
                generate_recipe(SAMPLE_INGREDIENTS, backend="gpt4all")


# ---------------------------------------------------------------------------
# _stub_generate
# ---------------------------------------------------------------------------

class TestStubGenerate:
    def test_returns_markdown_string(self):
        from llm.generate import _stub_generate
        text = _stub_generate(SAMPLE_INGREDIENTS)
        assert isinstance(text, str)
        assert "#" in text  # Markdown headers

    def test_uses_candidate_title(self):
        from llm.generate import _stub_generate
        text = _stub_generate(SAMPLE_INGREDIENTS, SAMPLE_CANDIDATES)
        assert "Soupe de légumes" in text

    def test_contains_demo_note(self):
        from llm.generate import _stub_generate
        text = _stub_generate(SAMPLE_INGREDIENTS)
        assert "démo" in text.lower() or "stub" in text.lower()
