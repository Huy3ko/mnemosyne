"""Regression test for #1177: _initialize_locked isolation ordering defect.

Ensures that when HERMES_HOME is absent from os.environ and profile home is delivered
via ContextVar override (set_hermes_home_override), the provider resolves profile
isolation using the configured hermes_home and settles canonical owner to the profile
name ('markets') instead of 'default'.
"""
from __future__ import annotations

import types
from pathlib import Path

import hermes_constants as hc
import pytest

import mnemosyne_hermes as m


@pytest.fixture
def agent_init(monkeypatch):
    assert hc.__file__ is not None
    agent_root = str(Path(hc.__file__).resolve().parent)
    monkeypatch.syspath_prepend(agent_root)
    import agent.agent_init as ai

    return ai


@pytest.mark.parametrize("isolation", ["true"])
def test_profile_home_identity_must_settle_on_profile(tmp_path, monkeypatch, agent_init, isolation):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / ".hermes" / "profiles" / "markets"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(f"memory:\n  mnemosyne:\n    profile_isolation: {isolation}\n")

    ids = {f"_{i}": None for i in agent_init._GATEWAY_IDENTITY_PARAMS}
    agent = types.SimpleNamespace(
        **ids,
        session_id="s",
        _session_db=None,
        session_cwd=None,
        _emit_warning=lambda *a, **k: None,
        _emit_status=lambda *a, **k: None,
    )

    tok = hc.set_hermes_home_override(str(home))
    try:
        kw = agent_init._memory_provider_init_kwargs(agent, "desktop")
        assert kw.get("agent_identity") == "markets"
        provider = m.MnemosyneMemoryProvider()
        provider.initialize(**kw)
        owner = provider._canonical_owner()
    finally:
        hc._HERMES_HOME_OVERRIDE.reset(tok)

    assert owner == "markets", f"owner settled to {owner!r}; identity lost at init"
