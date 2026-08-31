"""Failure-recovery tests for the long Section 4.1 orchestration script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_section_4_1 import JsonlAudit  # noqa: E402


def test_jsonl_audit_retries_transient_permission_error(tmp_path, monkeypatch):
    delays = []
    audit = JsonlAudit(
        tmp_path / "audit.jsonl", write_attempts=4, retry_delay_s=0.1,
        sleep=delays.append)
    real_append = audit._append_line
    calls = 0

    def fail_twice(line: str) -> None:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise PermissionError("synthetic transient lock")
        real_append(line)

    monkeypatch.setattr(audit, "_append_line", fail_twice)
    event = {"event": "step_accepted", "time_end_h": 1.0}
    audit(event)

    assert audit.events == [event]
    assert audit.write_retries == 2
    assert delays == [0.1, 0.2]
    rows = [json.loads(line) for line in audit.path.read_text().splitlines()]
    assert rows == [event]
