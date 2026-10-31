"""Ownership and session isolation for the routed banks (review round 2).

Two questions are settled here, both raised against the first fix round:

  O1  SQLite ownership.  The provider caches one ``Mnemosyne`` instance per
      bank, and every instance holds a live SQLite handle.  Handles are
      **thread-local** in the core: two threads that open the same file own
      two different connection objects, and two instances in the *same*
      thread share one object.  A shutdown that closes the handle must
      therefore (a) reach a handle that was opened in another thread, and
      (b) not break an instance that still uses it.

  O2  Session isolation.  A lifecycle operation (session end, auto-sleep,
      the sleep tool) must run against the bank of the session that
      triggered it — never against the bank of another session.

The first round replaced a remembered "last routed beam" pointer with a
route that is derived from the current session.  The pointer named
whichever chat wrote last, so a lifecycle operation of the *first* chat ran
against the *second* chat's bank.

The assertions compare the **exact** database path.  A substring check would
pass on the private profile storage as well: the profile directory is named
after the identity, so a path like ``homes/alice/mnemosyne/data/...``
contains "alice" without being Alice's bank.

The reproductions use synthetic temporary databases with the host LLM
disabled; no production data is read or written.
"""

import json
import sqlite3
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INTEGRATIONS_SRC = REPO_ROOT / "integrations" / "hermes" / "src"

ALICE_KEY = "agent:main:telegram:dm:111"
BOB_KEY = "agent:main:telegram:dm:222"


def _write_map(monkeypatch, data_dir: Path, mapping: dict) -> Path:
    """Write the map and pin the env to it.

    An ambient ``MNEMOSYNE_BANKS_MAP`` (the production deployment exports
    one) would otherwise decide the routing of a unit test.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "banks_map.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    monkeypatch.setenv("MNEMOSYNE_BANKS_MAP", str(path))
    return path


def _provider(tmp_path, monkeypatch, *, session_id, gateway_key, identity):
    """Build a provider whose banks live under ``tmp_path/data``."""
    from mnemosyne_hermes import MnemosyneMemoryProvider

    data = tmp_path / "data"
    (data / "private").mkdir(parents=True, exist_ok=True)
    hermes_home = tmp_path / "homes" / identity
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(hermes_home))
    provider = MnemosyneMemoryProvider()
    provider.initialize(
        session_id=session_id,
        hermes_home=str(hermes_home),
        agent_identity=identity,
        gateway_session_key=gateway_key,
        agent_context="primary",
    )
    return provider


def _call(provider, tool: str, args: dict | None = None) -> dict:
    return json.loads(provider.handle_tool_call(tool, args or {}))


def _remember(provider, content: str) -> dict:
    return _call(provider, "mnemosyne_remember", {"content": content, "scope": "global"})


def _recall_text(provider, query: str) -> str:
    payload = _call(provider, "mnemosyne_recall", {"query": query})
    return " ".join(str(r.get("content", "")) for r in payload.get("results", []))


def _bank_path(provider, bank: str) -> Path:
    """The exact database path of a bank."""
    from mnemosyne_hermes import bank_router

    return bank_router.bank_db_path(bank, getattr(provider, "_hermes_home", None))


def _bank_entry(provider, bank: str):
    """The cached ``(beam, mem)`` entry of a bank, or None."""
    from mnemosyne_hermes import bank_router

    cache = getattr(provider, "_bank_cache", None) or {}
    key = bank_router.cache_key(bank, getattr(provider, "_hermes_home", None))
    return cache.get(key)


def _capturing_beam(monkeypatch):
    """Install a fake Beam class that records the database it was built on.

    The session-end and auto-sleep workers build their own Beam instance from
    the resolved sleep arguments, so the class is the right place to observe
    which database the consolidation would run against.
    """
    import mnemosyne_hermes as provider_module

    captured: dict = {}

    class _FakeBeam:
        def __init__(self, **kwargs):
            captured.update({k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()})
            self.canonical_owner_id = "default"
            self.agent_context = "primary"

        def sleep(self):
            return None

    monkeypatch.setattr(provider_module, "_get_beam_class", lambda: _FakeBeam)
    return captured


def _capture_sleep_call(provider, bank: str, monkeypatch) -> dict:
    """Record the database the sleep *tool* consolidates.

    The sleep tool calls ``self._beam.sleep(...)`` on the Beam that the
    current scope bound, so the Beam instance itself is the observation point
    (the tool builds no new instance).
    """
    entry = _bank_entry(provider, bank)
    assert entry is not None, f"the bank {bank!r} was never bound"
    beam = entry[0]
    captured: dict = {}

    def _record(*args, **kwargs):
        captured["db_path"] = str(getattr(beam, "db_path", ""))
        return None

    monkeypatch.setattr(beam, "sleep", _record, raising=False)
    return captured


# --- O1: SQLite ownership --------------------------------------------------


def test_shutdown_must_close_a_handle_opened_in_another_thread(tmp_path, monkeypatch):
    """A handle opened in a worker thread must be closed by the shutdown.

    The core caches connections per thread.  A bank that was bound in a
    worker thread therefore owns a handle that the shutting-down thread does
    not have in its own cache — the release must reach the handle itself, not
    the thread-local cache of the caller.
    """
    _write_map(
        monkeypatch, tmp_path / "data",
        {"_default": "vela", "_unknown": "quarantine", ALICE_KEY: "alice"},
    )
    provider = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key=ALICE_KEY, identity="alice",
    )

    holder: dict = {}

    def _work_in_worker() -> None:
        with provider._beam_session_scope(""):
            pass
        entry = _bank_entry(provider, "alice")
        assert entry is not None, "the bank was not bound in the worker thread"
        holder["handle"] = entry[1].conn
        holder["thread"] = threading.current_thread()

    worker = threading.Thread(target=_work_in_worker, name="bank-worker")
    worker.start()
    worker.join(10)

    handle = holder["handle"]
    assert handle is not None, "the bank holds no connection"
    assert holder["thread"] is not threading.current_thread()

    # The handle works before the shutdown.
    assert handle.execute("SELECT 1").fetchone()[0] == 1

    provider.shutdown()

    with pytest.raises(sqlite3.ProgrammingError):
        handle.execute("SELECT 1")


def test_shutdown_must_not_break_a_sibling_provider_that_shares_the_bank(tmp_path, monkeypatch):
    """Two providers may share one bank handle — a shutdown must not kill it.

    The bank path is derived from the data directory and the bank name, so
    two providers in the same process open the *same* file.  In one thread
    that is one connection object, shared by both instances.  Closing it on
    the first provider's shutdown would break the second provider's live
    instance with a "cannot operate on a closed database" error.
    """
    _write_map(
        monkeypatch, tmp_path / "data",
        {"_default": "vela", "_unknown": "quarantine", ALICE_KEY: "alice"},
    )
    first = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key=ALICE_KEY, identity="alice",
    )
    second = _provider(
        tmp_path, monkeypatch,
        session_id="sess-c", gateway_key=ALICE_KEY, identity="alice",
    )

    assert _remember(first, "Satz aus der ersten Instanz").get("status") == "stored"
    assert _remember(second, "Satz aus der zweiten Instanz").get("status") == "stored"

    entry_first = _bank_entry(first, "alice")
    entry_second = _bank_entry(second, "alice")
    assert entry_first is not None and entry_second is not None
    assert entry_first[1].conn is entry_second[1].conn, (
        "the two providers do not share the handle — the test would prove nothing"
    )

    first.shutdown()

    # The second provider still owns the handle and must still work.
    assert "Satz aus der zweiten Instanz" in _recall_text(second, "zweiten Instanz"), (
        "the shutdown of another provider broke the shared bank connection"
    )

    second.shutdown()


def test_repeated_initialization_must_keep_one_connection_per_storage_path(tmp_path, monkeypatch):
    """A second ``initialize()`` must not multiply the live connections.

    The provider rebuilds its profile instance on every initialization.
    The bank cache is keyed by storage identity, so the same bank must
    still be served by one instance — and one handle — afterwards.
    """
    _write_map(
        monkeypatch, tmp_path / "data",
        {"_default": "vela", "_unknown": "quarantine", ALICE_KEY: "alice"},
    )
    provider = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key=ALICE_KEY, identity="alice",
    )
    assert _remember(provider, "Ein satz fuer Alice").get("status") == "stored"

    entry = _bank_entry(provider, "alice")
    assert entry is not None
    first_handle = entry[1].conn

    provider.initialize(
        session_id="sess-b",
        hermes_home=str(tmp_path / "homes" / "alice"),
        agent_identity="alice",
        gateway_session_key=ALICE_KEY,
        agent_context="primary",
    )

    entry_after = _bank_entry(provider, "alice")
    assert entry_after is not None, "the bank cache lost the bank on re-initialization"

    cache = getattr(provider, "_bank_cache", None) or {}
    handles = {id(value[1].conn) for value in cache.values()}
    paths = {value[1].db_path for value in cache.values()}
    assert len(handles) == len(paths), (
        f"one storage path is served by several connections: {len(handles)} "
        f"handles for {len(paths)} paths"
    )

    provider.shutdown()

    with pytest.raises(sqlite3.ProgrammingError):
        first_handle.execute("SELECT 1")


def test_shutdown_must_close_the_profile_connection_too(tmp_path, monkeypatch):
    """The profile instance owns a handle as well — it must be released.

    ``Mnemosyne`` has no ``close()`` method, so the provider's
    ``self._memory.close()`` raised and was swallowed.  The handle of the
    profile storage stayed open after the shutdown.
    """
    _write_map(
        monkeypatch, tmp_path / "data",
        {"_default": "vela", "_unknown": "quarantine", ALICE_KEY: "alice"},
    )
    provider = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key=ALICE_KEY, identity="alice",
    )
    assert _remember(provider, "Ein satz fuer Alice").get("status") == "stored"

    memory = getattr(provider, "_memory", None)
    beam = getattr(provider, "_beam", None)
    handles = []
    for owner in (memory, beam):
        handle = getattr(owner, "conn", None)
        if handle is not None and not any(handle is h for h in handles):
            handles.append(handle)
    assert handles, "the provider holds no profile connection"

    provider.shutdown()

    still_alive = []
    for handle in handles:
        try:
            handle.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            continue
        still_alive.append(handle)
    assert not still_alive, (
        f"{len(still_alive)} profile connections survived the shutdown"
    )


# --- O2: session isolation -------------------------------------------------


def _write_to_both_banks(provider) -> None:
    """Alice writes, then Bob writes. Both banks exist afterwards."""
    assert _remember(provider, "Satz von Alice").get("status") == "stored"
    provider.on_session_switch("sess-b", gateway_session_key=BOB_KEY)
    assert _remember(provider, "Satz von Bob").get("status") == "stored"


def test_session_end_must_use_the_current_session_bank(tmp_path, monkeypatch):
    """After a switch back, the session end must target that session's bank.

    The sticky pointer named whichever chat wrote last.  Ending Alice's
    session after Bob had written therefore consolidated *Bob's* bank.
    """
    _write_map(
        monkeypatch, tmp_path / "data",
        {"_default": "vela", "_unknown": "quarantine", ALICE_KEY: "alice", BOB_KEY: "bob"},
    )
    provider = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key=ALICE_KEY, identity="alice",
    )
    _write_to_both_banks(provider)

    # Back to Alice: her session ends now.
    provider.on_session_switch("sess-c", gateway_session_key=ALICE_KEY)

    captured = _capturing_beam(monkeypatch)
    monkeypatch.setattr(provider, "_reserve_reflection_budget_locked", lambda *a, **k: None)

    provider.on_session_end([{"role": "user", "content": "bis gleich"}])

    assert captured.get("db_path"), "the session-end sleep never ran"
    assert captured["db_path"] == str(_bank_path(provider, "alice")), (
        f"the consolidation ran against {captured['db_path']} instead of Alice's bank"
    )


def test_auto_sleep_must_use_the_current_session_bank(tmp_path, monkeypatch):
    """The auto-sleep of a turn must run against that turn's bank.

    With the sticky pointer the snapshot was built from the last routed
    write, so Alice's tenth turn consolidated Bob's bank — and the
    session-id guard rejected the snapshot instead, so Alice's backlog was
    never consolidated at all.
    """
    _write_map(
        monkeypatch, tmp_path / "data",
        {"_default": "vela", "_unknown": "quarantine", ALICE_KEY: "alice", BOB_KEY: "bob"},
    )
    provider = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key=ALICE_KEY, identity="alice",
    )
    _write_to_both_banks(provider)

    # Back to Alice — her turn triggers the auto-sleep.
    provider.on_session_switch("sess-c", gateway_session_key=ALICE_KEY)

    entry = _bank_entry(provider, "alice")
    assert entry is not None
    beam = entry[0]
    monkeypatch.setattr(beam, "get_working_stats", lambda: {"total": 5})
    monkeypatch.setattr(beam, "_count_unconsolidated_before", lambda cutoff: 5)
    provider._auto_sleep_threshold = 0

    captured = _capturing_beam(monkeypatch)
    monkeypatch.setattr(provider, "_reserve_reflection_budget_locked", lambda *a, **k: None)

    provider._maybe_auto_sleep(expected_session_id=provider._session_id)

    assert captured.get("db_path"), "the auto-sleep never ran"
    assert captured["db_path"] == str(_bank_path(provider, "alice")), (
        f"the auto-sleep ran against {captured['db_path']} instead of Alice's bank"
    )


def test_sleep_tool_must_use_the_current_session_bank(tmp_path, monkeypatch):
    """The sleep tool must consolidate the current session's bank."""
    _write_map(
        monkeypatch, tmp_path / "data",
        {"_default": "vela", "_unknown": "quarantine", ALICE_KEY: "alice", BOB_KEY: "bob"},
    )
    provider = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key=ALICE_KEY, identity="alice",
    )
    _write_to_both_banks(provider)

    provider.on_session_switch("sess-c", gateway_session_key=ALICE_KEY)

    captured = _capture_sleep_call(provider, "alice", monkeypatch)
    monkeypatch.setattr(provider, "_reserve_reflection_budget", lambda *a, **k: None)

    _call(provider, "mnemosyne_sleep", {})

    assert captured.get("db_path"), "the sleep tool never ran"
    assert captured["db_path"] == str(_bank_path(provider, "alice")), (
        f"the sleep tool ran against {captured['db_path']} instead of Alice's bank"
    )
