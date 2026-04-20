"""Sandbox tests.

Exercise the in-jail kernel via the SandboxRegistry. Marked as integration
since they spin a real subprocess. Skipped on platforms lacking bwrap AND
lacking /usr (we still run the rlimit-only fallback tests).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sandbox import SandboxRegistry, bwrap_works, ensure_sandbox_venv  # noqa: E402


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
VENV_DIR = PROJECT_DIR / ".sandbox-venv"
RUNNER = BACKEND_DIR / "sandbox_runner.py"


def _ensure_venv():
    # Keep test startup snappy: only install the absolute minimum.
    ensure_sandbox_venv(VENV_DIR, packages=["ipykernel"])


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    _ensure_venv()
    art = tmp_path_factory.mktemp("artifacts")
    reg = SandboxRegistry(
        artifacts_dir=art,
        venv_dir=VENV_DIR,
        runner_path=RUNNER,
        timeout_seconds=10,
        max_output_chars=2_000,
        idle_kill_minutes=60,
    )
    yield reg
    reg.shutdown_all()


def test_state_persists_across_calls(registry):
    sess = registry.get_or_create("conv-A", allow_network=False)
    r1 = sess.execute("x = 7")
    assert not r1["failed"], r1
    r2 = sess.execute("x * 6")
    assert not r2["failed"], r2
    assert r2["result"] == "42"


def test_last_expression_returns_repr(registry):
    sess = registry.get_or_create("conv-B", allow_network=False)
    r = sess.execute("import json\njson.dumps({'a': 1})")
    assert r["result"] == "'{\"a\": 1}'"


def test_stdout_captured(registry):
    sess = registry.get_or_create("conv-C", allow_network=False)
    r = sess.execute("print('hello'); 2+2")
    assert "hello" in r["stdout"]
    assert r["result"] == "4"


def test_timeout_kills_cell(registry):
    sess = registry.get_or_create("conv-D", allow_network=False)
    r = sess.execute("while True: pass", timeout=2)
    assert r["failed"]
    assert "Timeout" in r["stderr"] or "killed" in r["stderr"].lower()


def test_network_toggle_restarts_kernel(registry):
    sess1 = registry.get_or_create("conv-E", allow_network=False)
    sess1.execute("marker = 123")
    sess2 = registry.get_or_create("conv-E", allow_network=True)
    assert sess2 is not sess1 or not sess1.started
    # State cleared on restart
    r = sess2.execute("marker")
    assert r["failed"], "expected NameError after kernel restart"


def test_output_truncation(registry):
    sess = registry.get_or_create("conv-F", allow_network=False)
    r = sess.execute("print('x' * 10_000)")
    assert len(r["stdout"]) <= registry.max_output_chars + 200  # + truncation note
    assert "truncated" in r["stdout"]


@pytest.mark.skipif(not bwrap_works(), reason="bwrap not functional on this host (apparmor/userns)")
def test_bwrap_blocks_write_outside_workspace(tmp_path):
    _ensure_venv()
    art = tmp_path / "artifacts"
    art.mkdir()
    reg = SandboxRegistry(
        artifacts_dir=art,
        venv_dir=VENV_DIR,
        runner_path=RUNNER,
        timeout_seconds=5,
    )
    try:
        sess = reg.get_or_create("conv-jail", allow_network=False)
        # /etc is read-only bind; writing must fail.
        r = sess.execute("open('/etc/rogue', 'w').write('x')")
        assert r["failed"], "write to /etc should fail in bwrap jail"
        # Writing to /workspace must succeed.
        r2 = sess.execute("open('/workspace/ok.txt', 'w').write('hi'); 'ok'")
        assert not r2["failed"], r2
        assert (art / "ok.txt").read_text() == "hi"
    finally:
        reg.shutdown_all()


@pytest.mark.skipif(not bwrap_works(), reason="bwrap not functional on this host (apparmor/userns)")
def test_bwrap_network_isolation(tmp_path):
    _ensure_venv()
    art = tmp_path / "artifacts"
    art.mkdir()
    reg = SandboxRegistry(
        artifacts_dir=art,
        venv_dir=VENV_DIR,
        runner_path=RUNNER,
        timeout_seconds=5,
    )
    try:
        sess = reg.get_or_create("conv-net", allow_network=False)
        r = sess.execute(
            "import socket\n"
            "try:\n"
            "    s = socket.socket()\n"
            "    s.settimeout(2)\n"
            "    s.connect(('1.1.1.1', 80))\n"
            "    print('CONNECTED')\n"
            "except OSError as e:\n"
            "    print('BLOCKED:', type(e).__name__)\n"
        )
        assert "BLOCKED" in r["stdout"], r
        assert "CONNECTED" not in r["stdout"]
    finally:
        reg.shutdown_all()
