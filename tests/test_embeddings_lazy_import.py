"""Tests for lazy import self-healing in mnemosyne.core.embeddings.

Root cause (already proven): ``np`` and ``TextEmbedding`` are bound at import
time, and ``_FASTEMBED_AVAILABLE`` is evaluated once at import. When
``mnemosyne.core.embeddings`` is imported during a window where site-packages
is absent from ``sys.path`` (the Hermes plugin-loader import window), both
globals bind to None and ``available()`` returns False for the whole process
lifetime, even after site-packages returns.

These tests pin the fix: ``available()`` must re-attempt the import when the
globals are empty, so dense retrieval self-heals without a reload; a failing
import must log exactly one warning naming the failing module instead of
silently returning False.
"""
from __future__ import annotations

import builtins
import fnmatch
import importlib
import logging
import os
import sys

from mnemosyne.core import embeddings


def _site_package_entries():
    return [
        p for p in sys.path
        if fnmatch.fnmatch(os.path.basename(p.rstrip(os.sep)), "*-packages")
    ]


def _make_blocked_import(original_import):
    def _blocked_import(name, *args, **kwargs):
        if (
            name == "numpy" or name.startswith("numpy.")
            or name == "fastembed" or name.startswith("fastembed.")
        ):
            raise ModuleNotFoundError(
                f"No module named '{name}' (simulated site-packages import window)"
            )
        return original_import(name, *args, **kwargs)
    return _blocked_import


def _snapshot_path():
    return list(sys.path)


def _strip_site_packages(path):
    removed = []
    for p in _site_package_entries():
        if p in path:
            path.remove(p)
            removed.append(p)
    return removed


def test_available_self_heals_after_site_packages_returns(monkeypatch):
    for key in (
        "MNEMOSYNE_NO_EMBEDDINGS",
        "MNEMOSYNE_SKIP_EMBEDDINGS",
        "MNEMOSYNE_EMBEDDINGS_OFF",
    ):
        monkeypatch.delenv(key, raising=False)

    original_path = _snapshot_path()
    removed = _strip_site_packages(sys.path)
    assert removed, "expected at least one *-packages entry on sys.path"

    original_import = builtins.__import__
    try:
        # Simulate the plugin-loader import window: numpy/fastembed cannot be
        # imported while site-packages is off sys.path. We block the imports
        # rather than ejecting the modules from sys.modules: deleting already
        # loaded C extensions makes re-import fail with "cannot load module
        # more than once per process", which would mask the fix under test.
        builtins.__import__ = _make_blocked_import(original_import)
        importlib.reload(embeddings)

        assert embeddings.available() is False
        assert embeddings.np is None
        assert embeddings.TextEmbedding is None
        assert embeddings._FASTEMBED_AVAILABLE is False

        # site-packages returns: available() self-heals without a reload,
        # and behaviour no longer gates on the frozen import-time alias.
        sys.path[:] = original_path
        builtins.__import__ = original_import
        assert embeddings.available() is True
        assert embeddings.np is not None
        assert embeddings.TextEmbedding is not None
        assert embeddings._FASTEMBED_AVAILABLE is False  # frozen alias, unused
    finally:
        builtins.__import__ = original_import
        sys.path[:] = original_path
        importlib.reload(embeddings)


def test_import_failure_warns_exactly_once(monkeypatch, caplog):
    for key in (
        "MNEMOSYNE_NO_EMBEDDINGS",
        "MNEMOSYNE_SKIP_EMBEDDINGS",
        "MNEMOSYNE_EMBEDDINGS_OFF",
    ):
        monkeypatch.delenv(key, raising=False)

    original_path = _snapshot_path()
    removed = _strip_site_packages(sys.path)
    assert removed, "expected at least one *-packages entry on sys.path"

    original_import = builtins.__import__
    try:
        builtins.__import__ = _make_blocked_import(original_import)
        caplog.set_level(logging.WARNING, logger="mnemosyne.core.embeddings")
        importlib.reload(embeddings)
        for _ in range(5):
            embeddings.available()
    finally:
        builtins.__import__ = original_import

    records = [
        r for r in caplog.records
        if r.name == "mnemosyne.core.embeddings" and r.levelno >= logging.WARNING
    ]
    assert len(records) == 1, records
    msg = records[0].getMessage()
    assert "numpy" in msg or "fastembed" in msg, msg

    # Leave the module in a clean, bound state for sibling tests.
    sys.path[:] = original_path
    importlib.reload(embeddings)
