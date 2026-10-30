"""Polyphonic recall must give the same order for the same data and query (#1284).

String hashing is randomized per process, so iterating a set of entity names or memory ids
gives a different order in each run. Graph hits often tie (every gist scores 0.6), and RRF
fusion turns that order into ranks, so the final order changed between runs. These tests
run the same lookup under several PYTHONHASHSEED values and require identical output.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from mnemosyne.core.episodic_graph import EpisodicGraph, Gist, GraphEdge  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
HASH_SEEDS = range(8)
QUERY = "Alice met Bob, Carol, Dave and Erin"
PEOPLE = ["Alice", "Bob", "Carol", "Dave", "Erin"]

GRAPH_VOICE = """
import json, sys
from pathlib import Path
from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine
engine = PolyphonicRecallEngine(db_path=Path(sys.argv[1]))
print(json.dumps([
    [r.memory_id, r.voice, r.score, r.metadata.get("entity"), r.metadata.get("seed")]
    for r in engine._graph_voice(sys.argv[2])
]))
"""

PARTICIPANTS = """
import json, sys
from mnemosyne.core.episodic_graph import EpisodicGraph
print(json.dumps(EpisodicGraph._extract_participants(None, sys.argv[1])))
"""


@pytest.fixture
def graph_db(tmp_path):
    """One tied gist per person, and ctx edges where one node is reachable from several
    seeds and the second hop fans out, so both the seed loop and the BFS level order matter."""
    path = tmp_path / "graph.db"
    graph = EpisodicGraph(db_path=path)
    ts = "2026-01-01T00:00:00"
    for name in PEOPLE:
        graph.store_gist(
            Gist(f"gist_{name}", f"{name} was there", ts, [name], None, None, None), name
        )
        graph.add_edge(GraphEdge(name, "shared", "ctx", 0.9, ts))
        for i in range(4):
            graph.add_edge(GraphEdge(name, f"{name}-hop1-{i}", "ctx", 0.9, ts))
            graph.add_edge(GraphEdge(f"{name}-hop1-{i}", f"{name}-hop2-{i}", "ctx", 0.9, ts))
    graph.add_edge(GraphEdge("Alice-hop1-0", "Bob-hop2-1", "ctx", 0.9, ts))
    graph.conn.close()
    return path


def _run(seed, script, *args):
    env = dict(os.environ, PYTHONHASHSEED=str(seed), PYTHONPATH=str(REPO_ROOT))
    env.pop("MNEMOSYNE_VOICE_GRAPH", None)
    out = subprocess.run(
        [sys.executable, "-c", script, *args],
        env=env, cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def test_graph_voice_order_does_not_depend_on_hash_seed(graph_db):
    runs = {seed: _run(seed, GRAPH_VOICE, str(graph_db), QUERY) for seed in HASH_SEEDS}

    assert len(runs[0]) > len(PEOPLE)
    for seed, run in runs.items():
        assert run == runs[0], f"PYTHONHASHSEED={seed} differs from 0"



def test_gist_participants_do_not_depend_on_hash_seed():
    """The participant list is capped at 5, so with more candidates the dedupe order
    decides which names are stored on the gist, and so which gists the graph voice finds."""
    text = "Alice asked Bob and Carol, then Dave told Erin and Frank that we agreed"
    runs = {seed: _run(seed, PARTICIPANTS, text) for seed in HASH_SEEDS}

    assert runs[0] == ["Alice", "Bob", "Carol", "Dave", "Erin"]
    for seed, run in runs.items():
        assert run == runs[0], f"PYTHONHASHSEED={seed} differs from 0"
