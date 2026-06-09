"""Tests for the model router — the single source of truth for model selection."""

import importlib

from polymath.models.router import Role, model_for, models_for


def test_every_role_has_a_model():
    for role in Role:
        assert model_for(role), f"no model configured for {role}"


def test_models_for_starts_with_primary_and_adds_fallbacks():
    for role in Role:
        chain = models_for(role)
        assert chain[0] == model_for(role)  # primary first
        assert len(chain) > 1  # has fallbacks
        assert len(chain) == len(set(chain))  # no duplicates


def test_models_for_env_override_is_primary():
    import os

    os.environ["POLYMATH_MODEL_WRITER"] = "custom/model"
    try:
        import polymath.models.router as router

        importlib.reload(router)
        assert router.models_for(router.Role.WRITER)[0] == "custom/model"
    finally:
        del os.environ["POLYMATH_MODEL_WRITER"]
        importlib.reload(router)


def test_search_role_default():
    # Search role must use a free-tier model variant.
    assert model_for(Role.SEARCH).endswith(":free")


def test_env_override(monkeypatch):
    monkeypatch.setenv("POLYMATH_MODEL_WRITER", "some/custom-model")
    import polymath.models.router as router

    importlib.reload(router)
    assert router.model_for(router.Role.WRITER) == "some/custom-model"
