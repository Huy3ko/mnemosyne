"""Bank-Routing fuer den Hermes-Mnemosyne-Adapter.

Eine Vela, getrennte Erinnerungsbanken. Der Adapter bindet nativ eine Bank pro
Profil; hier wird daraus eine Bindung pro Gespraechspartner, abgeleitet aus der
**Plattform-Identitaet** des Turns (Session-Key / Chat-ID) — niemals aus dem
Gespraechsinhalt.

Regeln (fail-closed):

1. Ein Eintrag in `banks_map.json` entscheidet: laengster Praefix des
   Session-Keys gewinnt.
2. Kein Key vorhanden (CLI, Cron, interner Lauf ohne Chat) -> `_default`.
3. Key vorhanden, aber unbekannt (fremder Chat, Gruppe) -> `_unknown`
   (Default `quarantine`). Eine unbekannte Zuordnung darf niemals eine private
   Bank oeffnen.
4. Ist die Bank nicht lesbar/oeffenbar, wirft der Aufrufer — er faellt **nicht**
   auf die private Bank zurueck.

Die Zuordnung ist damit Daten, nicht Code: eine neue Person ist ein Eintrag,
keine Aenderung am Adapter.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

MAP_FILENAME = "banks_map.json"
_ENV_MAP = "MNEMOSYNE_BANKS_MAP"
_lock = threading.Lock()
_cache: dict[str, tuple[float, dict]] = {}


def data_dir(hermes_home: Optional[str] = None) -> Path:
    """Datenverzeichnis der Mnemosyne-Banken."""
    explicit = os.environ.get("MNEMOSYNE_DATA_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    if hermes_home:
        return Path(hermes_home).expanduser() / "mnemosyne" / "data"
    return Path.home() / ".hermes" / "mnemosyne" / "data"


def map_path(hermes_home: Optional[str] = None) -> Path:
    explicit = os.environ.get(_ENV_MAP, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return data_dir(hermes_home) / MAP_FILENAME


def load_map(hermes_home: Optional[str] = None) -> dict:
    """Zuordnung laden (mit mtime-Cache; Datei darf zur Laufzeit geaendert werden)."""
    path = map_path(hermes_home)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    key = str(path)
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    entries = {str(k): str(v) for k, v in data.items() if not str(k).startswith("_")}
    with _lock:
        _cache[key] = (mtime, entries)
    return entries


def _meta(key: str, field: str, fallback: str) -> str:
    """Metafelder (`_default`, `_unknown`) aus derselben Datei lesen."""
    path = map_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback
    value = raw.get(field) if isinstance(raw, dict) else None
    return str(value) if value else fallback


def is_chat_identity(key: str) -> bool:
    """Sieht dieser Key wie eine Chat-Identitaet aus dem Gateway aus?

    Format aus `gateway/session.py:build_session_key`:
    `<ns>:<platform>:<chat_type>[...]` mit `ns` = `agent:<profil>`.
    Eine reine Zustands-/Lauf-ID (Cron, Subagent, interner Lauf) ist **keine**
    Chat-Identitaet. Sie darf nicht als "unbekannter Chat" in Quarantaene fallen,
    denn sie gehoert keiner Person und keine fremde Person liest sie.
    """
    parts = str(key or "").split(":")
    # <ns> ist selbst zweiteilig ("agent:<profil>"), danach Plattform und Chat-Typ.
    if len(parts) < 4 or parts[0] != "agent" or not parts[1]:
        return False
    return parts[3] in {"dm", "group", "channel", "thread"}


def resolve_bank(session_key: str, hermes_home: Optional[str] = None) -> tuple[str, str]:
    """(bank, grund) fuer diesen Turn. Wirft nie; liefert immer eine Bank."""
    key = str(session_key or "").strip()
    entries = load_map(hermes_home)

    if not key:
        bank = _meta(key, "_default", "quarantine")
        return bank, "default:kein-session-key"

    if not is_chat_identity(key):
        bank = _meta(key, "_default", "quarantine")
        return bank, "default:kein-chat-kontext"

    best = ""
    best_bank = ""
    for pattern, bank in entries.items():
        if key.startswith(pattern) and len(pattern) > len(best):
            best, best_bank = pattern, bank
    if best:
        return best_bank, f"map:{best}"

    bank = _meta(key, "_unknown", "quarantine")
    return bank, "unknown:fail-closed"


def bank_db_path(bank: str, hermes_home: Optional[str] = None) -> Path:
    return data_dir(hermes_home) / "banks" / bank / "mnemosyne.db"
