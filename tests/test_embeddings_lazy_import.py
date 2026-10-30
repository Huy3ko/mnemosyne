"""Tests for lazy import self-healing in mnemosyne.core.embeddings.

Root cause (already proven): ``np`` and ``TextEmbedding`` are bound at import
time, and ``_FASTEMBED_AVAILABLE`` is evaluated once at import. When
``mnemosyne.core.embeddings`` is imported during a window where the dense stack
is unreachable (the Hermes plugin-loader import window, or a venv without the
embeddings extra), both globals bind to None and ``available()`` returns False
for the whole process lifetime, even after the stack becomes importable.

These tests pin the fix: ``available()`` must re-attempt the import when the
globals are empty, so dense retrieval self-heals without a reload; a failing
import must log exactly one warning naming the failing module instead of
silently returning False.

The tests install stub ``numpy``/``fastembed`` modules, so they do NOT require
the optional ``embeddings`` extra to run — only the retry behaviour is under
test here, not real embedding quality.
"""
from __future__ import annotations

import builtins
import fnmatch
import importlib
import logging
import os
import sys
import types

from mnemosyne.core import embeddings

_DENSE_MODULES = ("numpy", "fastembed")


def _site_package_entries():
    return [
        p for p in sys.path
        if fnmatch.fnmatch(os.path.basename(p.rstrip(os.sep)), "*-packages")
    ]


def _make_blocked_import(original_import):
    def _blocked_import(name, *args, **kwargs):
        if name in _DENSE_MODULES or name.split(".")[0] in _DENSE_MODULES:
            raise ModuleNotFoundError(
                f"No module named '{name}' (simulated dense-stack import window)"
            )
        return original_import(name, *args, **kwargs)
    return _blocked_import


def _install_stub_dense_stack():
    """Register stub ``numpy``/``fastembed`` modules and return the previous entries.

    Using stubs keeps the recovery assertion independent of whether the optional
    ``embeddings`` extra is installed, which is the state a CI run without it is
    in. The previous entries (possibly None) are restored by the caller.
    """
    np_stub = types.ModuleType("numpy")
    setattr(np_stub, "__version__", "0.0.0-stub")

    fastembed_stub = types.ModuleType("fastembed")

    class TextEmbedding:  # noqa: N801 - mirrors the real export name
        def __init__(self, *args, **kwargs):
            pass

    setattr(fastembed_stub, "TextEmbedding", TextEmbedding)

    previous = {name: sys.modules.get(name) for name in _DENSE_MODULES}
    sys.modules["numpy"] = np_stub
    sys.modules["fastembed"] = fastembed_stub
    return previous, (np_stub, TextEmbedding)


def _restore_stub_dense_stack(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_available_self_heals_after_dense_stack_returns(monkeypatch):
    for key in (
        "MNEMOSYNE_NO_EMBEDDINGS",
        "MNEMOSYNE_SKIP_EMBEDDINGS",
        "MNEMOSYNE_EMBEDDINGS_OFF",
    ):
        monkeypatch.delenv(key, raising=False)

    original_path = list(sys.path)
    original_import = builtins.__import__
    previous_modules = None
    try:
        # Simulate the import window: the dense stack cannot be imported at all.
        # Blocking the import (rather than ejecting loaded C extensions) keeps
        # re-import safe: deleting an already-loaded extension makes a later
        # import fail with "cannot load module more than once per process".
        builtins.__import__ = _make_blocked_import(original_import)
        importlib.reload(embeddings)

        assert embeddings.available() is False
        assert embeddings.np is None
        assert embeddings.TextEmbedding is None
        assert embeddings._FASTEMBED_AVAILABLE is False

        # The stack becomes importable again. With the real extra installed the
        # bindings come from it; without it (CI, minimal install) the stubs
        # stand in. Either way the retry must happen without a reload.
        previous_modules, (stub_np, stub_text_embedding) = _install_stub_dense_stack()
        builtins.__import__ = original_import

        assert embeddings.available() is True
        assert embeddings.np is stub_np
        assert embeddings.TextEmbedding is stub_text_embedding
        assert embeddings._FASTEMBED_AVAILABLE is False  # frozen alias, unused
    finally:
        builtins.__import__ = original_import
        sys.path[:] = original_path
        if previous_modules is not None:
            _restore_stub_dense_stack(previous_modules)
        importlib.reload(embeddings)


def test_import_failure_warns_exactly_once(monkeypatch, caplog):
    for key in (
        "MNEMOSYNE_NO_EMBEDDINGS",
        "MNEMOSYNE_SKIP_EMBEDDINGS",
        "MNEMOSYNE_EMBEDDINGS_OFF",
    ):
        monkeypatch.delenv(key, raising=False)

    original_path = list(sys.path)
    original_import = builtins.__import__
    try:
        builtins.__import__ = _make_blocked_import(original_import)
        caplog.set_level(logging.WARNING, logger="mnemosyne.core.embeddings")
        importlib.reload(embeddings)
        # Call repeatedly to prove the warning budget is per-process, not per-call.
        results = [embeddings.available() for _ in range(5)]
        assert results == [False] * 5
    finally:
        builtins.__import__ = original_import
        sys.path[:] = original_path

    records = [
        r for r in caplog.records
        if r.name == "mnemosyne.core.embeddings" and r.levelno >= logging.WARNING
    ]
    assert len(records) == 1, records
    msg = records[0].getMessage()
    assert "numpy" in msg or "fastembed" in msg, msg

    # Leave the module in a clean, bound state for sibling tests.
    importlib.reload(embeddings)
