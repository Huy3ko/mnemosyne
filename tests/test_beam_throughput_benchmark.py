"""Contract tests for the advisory harness; deliberately no timing thresholds."""
import copy
import json
import os
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest

from _benchmarks import bench_beam_throughput as bench


def test_summary_uses_nearest_rank_and_whole_round_wall_time():
    batches = [[i * 1_000_000 for i in range(1, 11)],
               [i * 1_000_000 for i in range(11, 21)]]
    result = bench.summarize(batches, [100_000_000, 200_000_000])
    assert result["calls"] == 20
    assert result["latency_ms"] == {
        "mean": 10.5, "p50": 10, "p95": 19, "min": 1, "max": 20,
    }
    assert result["round_qps"] == [100, 50]
    assert result["median_qps"] == 75


def test_measure_excludes_warmup_and_cycles_fixed_queries(monkeypatch):
    calls = []

    class FakeBeam:
        def recall(self, query, **kwargs):
            calls.append((query, kwargs))
            return [{"id": "synthetic"}]

    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr(bench.time, "perf_counter_ns", lambda: next(clock))
    args = SimpleNamespace(warmup_cycles=2, rounds=2, cycles_per_round=1, top_k=3)
    samples, counts, elapsed = bench.measure(FakeBeam(), args)
    assert [q for q, _ in calls] == list(bench.QUERIES) * 4
    assert all(kwargs == {"top_k": 3, **bench.RECALL_KWARGS} for _, kwargs in calls)
    assert samples == [[10] * 8] * 2
    assert counts == [[1] * 8] * 2
    assert elapsed == [170, 170]


def test_empty_recall_refuses_to_publish_timing():
    beam = SimpleNamespace(recall=lambda *args, **kwargs: [])
    args = SimpleNamespace(warmup_cycles=1, rounds=1, cycles_per_round=1, top_k=3)
    with pytest.raises(RuntimeError, match="Empty recall"):
        bench.measure(beam, args)


def test_seed_has_fixed_corpus_and_no_ingestion_side_effects():
    snapshots = []
    for _ in range(2):
        conn = sqlite3.connect(":memory:")
        try:
            for table in ("working_memory", "episodic_memory"):
                conn.execute(f"CREATE TABLE {table} (id, content, source, timestamp, "
                             "session_id, importance, veracity)")
            bench.seed(SimpleNamespace(conn=conn), 16)
            rows = {
                table: conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
                for table in ("working_memory", "episodic_memory")
            }
            assert all(len(batch) == 16 for batch in rows.values())
            assert rows["working_memory"][0] == (
                "working_memory-00000000",
                "zephyr0 calibration ledger item 00000000 pressure valve inspection record for working_memory",
                "benchmark", "2026-01-01T12:00:00", "throughput", 0.4, "stated",
            )
            snapshots.append(rows)
        finally:
            conn.close()
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize("option", [
    "--rows-per-tier", "--top-k", "--warmup-cycles", "--cycles-per-round", "--rounds",
])
@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_cli_rejects_invalid_counts(option, value):
    with pytest.raises(SystemExit) as exc:
        bench.parser().parse_args(["--machine", "test", option, value])
    assert exc.value.code == 2


def test_real_cli_isolated_linear_report_and_baseline(tmp_path):
    ambient = tmp_path / "user-data"
    ambient.mkdir()
    config = ambient / "config.yaml"
    config.write_text("polyphonic_recall: true\nrecall_content_cap: 1\n", encoding="utf-8")
    output = tmp_path / "reports" / "baseline.json"
    command = [
        sys.executable, str(bench.ROOT / "_benchmarks" / "bench_beam_throughput.py"),
        "--machine", "pytest-machine", "--rows-per-tier", "16",
        "--warmup-cycles", "1", "--rounds", "2", "--cycles-per-round", "1",
        "--output", str(output),
    ]
    env = {**os.environ,
           "MNEMOSYNE_DATA_DIR": str(ambient),
           "MNEMOSYNE_POLYPHONIC_RECALL": "1",
           "MNEMOSYNE_RECALL_CONTENT_CAP": "not-an-integer",
           "MNEMOSYNE_EMBEDDINGS_VIA_API": "1",
           "MNEMOSYNE_EMBEDDING_API_KEY": "must-not-appear-in-report",
           "PYTHONHASHSEED": "random"}
    result = subprocess.run(command, env=env, capture_output=True, text=True,
                            encoding="utf-8", timeout=60, check=True)
    report = json.loads(result.stdout)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert "must-not-appear-in-report" not in result.stdout
    assert list(ambient.iterdir()) == [config]
    assert config.read_text(encoding="utf-8") == "polyphonic_recall: true\nrecall_content_cap: 1\n"
    assert report["validation"]["engine"] == "linear"
    assert report["validation"]["seeded_rows"] == {"working_memory": 16, "episodic_memory": 16}
    assert report["settings"] == {**bench.SETTINGS, "PYTHONHASHSEED": "0"}
    assert report["workload"]["queries"] == list(bench.QUERIES)
    assert len(report["samples_ns"]) == 2
    assert all(len(batch) == 8 for batch in report["samples_ns"])
    assert report["summary"]["calls"] == 16
    assert all(0 < count <= 10 for batch in report["result_counts"] for count in batch)
    assert report["summary"] == bench.summarize(report["samples_ns"], report["round_elapsed_ns"])
    assert report["environment"]["machine_label"] == "pytest-machine"
    assert len(report["revision"]["commit"]) == 40

    second = subprocess.run(command + ["--baseline", str(output)], env=env,
                            capture_output=True, text=True, encoding="utf-8",
                            timeout=60, check=True)
    compared = json.loads(second.stdout)
    assert compared["comparison"]["compatible"] is True
    assert "median_qps_change_pct" in compared["comparison"]

    # A slower compatible report is still advisory, never a failed timing gate.
    slower = copy.deepcopy(report)
    slower["summary"]["median_qps"] /= 2
    assert bench.compare(slower, report)["median_qps_change_pct"] == -50
    for field in ("workload", "environment", "settings", "validation", "sqlite_pragmas",
                  "benchmark", "harness_sha256"):
        changed = copy.deepcopy(report)
        changed[field] = None
        assert bench.compare(report, changed) == {
            "compatible": False, "mismatched_fields": [field],
        }
