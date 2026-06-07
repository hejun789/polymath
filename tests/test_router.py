"""Tests for the model router — the single source of truth for model selection."""

import importlib

from polymath.models.router import Role, model_for


def test_every_role_has_a_model():
    for role in Role:
        assert model_for(role), f"no model configured for {role}"


def test_search_role_default():
    assert model_for(Role.SEARCH) == "google/gemini-2.0-flash-exp:free"


def test_env_override(monkeypatch):
    monkeypatch.setenv("POLYMATH_MODEL_WRITER", "some/custom-model")
    import polymath.models.router as router

    importlib.reload(router)
    assert router.model_for(router.Role.WRITER) == "some/custom-model"
