"""Host-side sandbox manager for Python code execution.

Architecture:
  - One persistent Python subprocess per conversation, launched inside a
    `bwrap` jail with:
        * /usr, /etc, /bin, /lib, /lib64      → read-only bind
        * artifacts/                           → read-write bind at /workspace
        * <sandbox-venv>                       → read-only bind at /venv
        * /tmp, /home                          → tmpfs (wiped each launch)
        * user / pid / ipc / uts namespaces   → unshared
        * net namespace                        → unshared UNLESS allow_network=True
  - Cells communicate via JSON-over-stdin/stdout using `sandbox_runner.py`.
  - Kernel-level state persists across cells within a conversation.
  - Toggling network requires kernel restart (in-memory state cleared; files
    in artifacts/ persist).
  - Idle kill after N minutes; explicit shutdown on conversation delete.

If `bwrap` is not available on the host, the module falls back to a
resource-limit-only mode (rlimits + timeouts + output caps) and emits a
visible warning on first use. Writes outside artifacts/ are NOT prevented in
that fallback mode.
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


MARKER = "<<<SANDBOX>>>"

DEFAULT_PACKAGES = [
    "ipykernel",
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "openpyxl",
    "matplotlib",
    "requests",
]


# ── bwrap detection ───────────────────────────────────────────────────────────

def bwrap_path() -> Optional[str]:
    return shutil.which("bwrap")


_BWRAP_WORKS_CACHE: Optional[bool] = None


def bwrap_works() -> bool:
    """Test whether bwrap can actually launch a user namespace on this host.

    AppArmor on newer Ubuntu/Debian kernels sometimes blocks unprivileged user
    namespaces, making `bwrap --unshare-user` fail at startup. When that
    happens we must fall back to the rlimit-only path.
    """
    global _BWRAP_WORKS_CACHE
    if _BWRAP_WORKS_CACHE is not None:
        return _BWRAP_WORKS_CACHE
    bw = bwrap_path()
    if bw is None:
        _BWRAP_WORKS_CACHE = False
        return False
    try:
        r = subprocess.run(
            [bw, "--unshare-user", "--unshare-net", "--ro-bind", "/usr", "/usr",
             "--ro-bind", "/bin", "/bin", "--proc", "/proc", "--dev", "/dev",
             "--tmpfs", "/tmp", "/bin/true"],
            capture_output=True, timeout=5,
        )
        _BWRAP_WORKS_CACHE = r.returncode == 0
    except Exception:
        _BWRAP_WORKS_CACHE = False
    return _BWRAP_WORKS_CACHE


# ── Sandbox venv management ──────────────────────────────────────────────────

def ensure_sandbox_venv(venv_dir: Path, packages: list[str] = None, log=print) -> None:
    """Create the sandbox venv (idempotent). Blocks until ready.

    Installs the default data-science toolchain so the LLM can `import pandas`
    etc. without needing network access at runtime.
    """
    packages = packages or DEFAULT_PACKAGES
    venv_dir = Path(venv_dir)
    venv_python = venv_dir / "bin" / "python"

    if not venv_python.exists():
        log(f"[sandbox] creating venv at {venv_dir}")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])

    marker = venv_dir / ".pkgs-installed"
    expected = sorted(set(packages))
    if marker.exists() and marker.read_text().strip() == ",".join(expected):
        return

    log(f"[sandbox] installing packages into sandbox venv: {expected}")
    subprocess.check_call(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        stdout=subprocess.DEVNULL,
    )
    subprocess.check_call(
        [str(venv_python), "-m", "pip", "install", *expected],
    )
    marker.write_text(",".join(expected))
    log("[sandbox] venv ready")


# ── Build bwrap command ───────────────────────────────────────────────────────

def _bwrap_cmd(
    *,
    artifacts_dir: Path,
    venv_dir: Path,
    runner_path: Path,
    allow_network: bool,
    rlimit_as_bytes: int,
) -> list[str]:
    """Assemble the bwrap command line that launches sandbox_runner.py."""
    cmd = [
        "bwrap",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--ro-bind", "/etc", "/etc",
        "--bind", str(artifacts_dir), "/workspace",
        "--ro-bind", str(venv_dir), "/venv",
        "--ro-bind", str(runner_path), "/runner.py",
        "--tmpfs", "/tmp",
        "--tmpfs", "/home",
        "--dev", "/dev",
        "--proc", "/proc",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--die-with-parent",
        "--chdir", "/workspace",
        "--setenv", "PYTHONUNBUFFERED", "1",
        "--setenv", "PATH", "/venv/bin:/usr/bin:/bin",
        "--setenv", "HOME", "/home",
        "--setenv", "TMPDIR", "/tmp",
    ]
    if not allow_network:
        cmd.append("--unshare-net")
    # Some /lib64 distros don't have /lib64 as a symlink target; also bind it
    # if it exists as a real directory on host.
    if Path("/lib64").is_dir() and not Path("/lib64").is_symlink():
        # Replace the earlier --symlink with a ro-bind.
        idx = cmd.index("--symlink")
        del cmd[idx:idx + 3]
        cmd[idx:idx] = ["--ro-bind", "/lib64", "/lib64"]

    cmd += ["/venv/bin/python", "-u", "/runner.py"]
    return cmd


# ── Fallback (no bwrap): subprocess with rlimits ─────────────────────────────

def _rlimit_preexec(rlimit_as_bytes: int, rlimit_cpu_seconds: int):
    def _pre():
        resource.setrlimit(resource.RLIMIT_AS, (rlimit_as_bytes, rlimit_as_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (rlimit_cpu_seconds, rlimit_cpu_seconds))
        resource.setrlimit(resource.RLIMIT_NPROC, (200, 200))
        resource.setrlimit(resource.RLIMIT_FSIZE, (100 * 1024 * 1024, 100 * 1024 * 1024))
    return _pre


# ── Kernel session ────────────────────────────────────────────────────────────

@dataclass
class KernelSession:
    conversation_id: str
    artifacts_dir: Path
    venv_dir: Path
    runner_path: Path
    allow_network: bool
    timeout_seconds: int = 30
    max_output_chars: int = 10_000
    mode: str = "bwrap"                        # "bwrap" | "rlimit_only"
    _proc: Optional[subprocess.Popen] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: float = field(default_factory=time.time)
    started: bool = False

    def start(self) -> None:
        if self.started:
            return
        if self.mode == "bwrap":
            cmd = _bwrap_cmd(
                artifacts_dir=self.artifacts_dir,
                venv_dir=self.venv_dir,
                runner_path=self.runner_path,
                allow_network=self.allow_network,
                rlimit_as_bytes=1024 * 1024 * 1024,
            )
            preexec = None
        else:
            cmd = [
                str(self.venv_dir / "bin" / "python"), "-u", str(self.runner_path),
            ]
            preexec = _rlimit_preexec(1024 * 1024 * 1024, 120)

        env = {
            "PATH": f"{self.venv_dir / 'bin'}:/usr/bin:/bin",
            "PYTHONUNBUFFERED": "1",
            "HOME": "/home" if self.mode == "bwrap" else str(self.artifacts_dir),
            "MPLBACKEND": "Agg",  # non-interactive backend; no display needed
        }
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.artifacts_dir) if self.mode != "bwrap" else None,
            text=True,
            bufsize=1,
            preexec_fn=preexec,
            env=env if self.mode != "bwrap" else None,
            start_new_session=True,
        )

        # Drain the ready marker (first prefixed line).
        ready = self._read_marker_line(startup_timeout=20)
        if ready is None or not ready.get("ready"):
            raise RuntimeError(
                f"sandbox kernel failed to signal ready. stderr={self._drain_stderr()[:500]}"
            )
        self.started = True

    def _read_marker_line(self, startup_timeout: float = 20.0) -> Optional[dict]:
        """Read lines from stdout until we hit one starting with MARKER."""
        assert self._proc is not None and self._proc.stdout is not None
        deadline = time.time() + startup_timeout
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    return None
                continue
            if line.startswith(MARKER):
                try:
                    return json.loads(line[len(MARKER):].strip())
                except json.JSONDecodeError:
                    return None
        return None

    def _drain_stderr(self, limit: int = 2000) -> str:
        if not self._proc or not self._proc.stderr:
            return ""
        # Non-blocking read via os-level fd
        import fcntl
        fd = self._proc.stderr.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            data = self._proc.stderr.read(limit) or ""
        except Exception:
            data = ""
        return data

    def execute(self, code: str, timeout: Optional[int] = None) -> dict:
        """Run `code` in the kernel. Blocks until the cell returns or times out."""
        with self._lock:
            if not self.started:
                self.start()

            assert self._proc is not None
            t = int(timeout or self.timeout_seconds)
            self.last_used = time.time()

            req = {"id": str(uuid.uuid4()), "code": code, "timeout": t}
            try:
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                # Kernel died — surface a clear error; caller can restart.
                self.shutdown()
                return {
                    "stdout": "",
                    "stderr": f"sandbox kernel died while sending request: {e}",
                    "result": "",
                    "failed": True,
                    "kernel_died": True,
                }

            # Allow up to timeout + 5s for response; the in-jail SIGALRM raises
            # at `t` so the runner should respond promptly.
            host_deadline = time.time() + t + 5
            resp: Optional[dict] = None
            while time.time() < host_deadline:
                remaining = host_deadline - time.time()
                line = _readline_with_timeout(self._proc.stdout, min(1.0, max(0.1, remaining)))
                if line is None:
                    if self._proc.poll() is not None:
                        break
                    continue
                if line.startswith(MARKER):
                    try:
                        resp = json.loads(line[len(MARKER):].strip())
                        break
                    except json.JSONDecodeError:
                        continue

            if resp is None:
                # Hard kill — the cell wedged past SIGALRM (e.g. C-level loop).
                self.shutdown()
                return {
                    "stdout": "",
                    "stderr": (
                        f"cell did not respond within {t + 5}s even after SIGALRM; "
                        "kernel was killed. In-memory state lost; files in artifacts/ preserved."
                    ),
                    "result": "",
                    "failed": True,
                    "kernel_died": True,
                }

            # Truncate oversize output.
            def _cap(s: str) -> str:
                if len(s) <= self.max_output_chars:
                    return s
                return s[: self.max_output_chars] + f"\n[… {len(s) - self.max_output_chars} chars truncated …]"
            resp["stdout"] = _cap(resp.get("stdout", "") or "")
            resp["stderr"] = _cap(resp.get("stderr", "") or "")
            resp["result"] = _cap(resp.get("result", "") or "")
            return resp

    def shutdown(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
        self.started = False


def _readline_with_timeout(stream, timeout: float) -> Optional[str]:
    """Read one line from `stream` with a timeout. Returns None on timeout."""
    import select
    if stream is None:
        return None
    r, _, _ = select.select([stream], [], [], timeout)
    if not r:
        return None
    return stream.readline()


# ── Registry ──────────────────────────────────────────────────────────────────

@dataclass
class SandboxRegistry:
    artifacts_dir: Path
    venv_dir: Path
    runner_path: Path
    timeout_seconds: int = 30
    max_output_chars: int = 10_000
    idle_kill_minutes: int = 10
    _sessions: dict[str, KernelSession] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get_or_create(self, conversation_id: str, allow_network: bool) -> KernelSession:
        with self._lock:
            sess = self._sessions.get(conversation_id)
            if sess is not None and sess.allow_network != allow_network:
                sess.shutdown()
                sess = None
            if sess is None:
                sess = KernelSession(
                    conversation_id=conversation_id,
                    artifacts_dir=self.artifacts_dir,
                    venv_dir=self.venv_dir,
                    runner_path=self.runner_path,
                    allow_network=allow_network,
                    timeout_seconds=self.timeout_seconds,
                    max_output_chars=self.max_output_chars,
                    mode="bwrap" if bwrap_works() else "rlimit_only",
                )
                self._sessions[conversation_id] = sess
            return sess

    def shutdown_conversation(self, conversation_id: str) -> None:
        with self._lock:
            sess = self._sessions.pop(conversation_id, None)
            if sess:
                sess.shutdown()

    def shutdown_all(self) -> None:
        with self._lock:
            for sess in list(self._sessions.values()):
                sess.shutdown()
            self._sessions.clear()

    def reap_idle(self) -> list[str]:
        """Kill sessions idle longer than idle_kill_minutes. Returns killed conv ids."""
        killed = []
        threshold = time.time() - self.idle_kill_minutes * 60
        with self._lock:
            for cid in list(self._sessions.keys()):
                sess = self._sessions[cid]
                if sess.last_used < threshold:
                    sess.shutdown()
                    del self._sessions[cid]
                    killed.append(cid)
        return killed
