"""In-jail Python execution loop.

Runs inside the bwrap sandbox. Reads JSON-encoded execute requests from stdin,
runs each one against a persistent globals dict (Jupyter-style state across
calls), captures stdout/stderr + the last-expression value, and writes a
JSON-encoded response to stdout.

Protocol:
  Request  (one JSON object per line on stdin):
      {"id": "<call-id>", "code": "...", "timeout": 30}
  Response (one JSON object per line on stdout, prefixed with "<<<SANDBOX>>>"
  so host-side parsing can ignore anything the user's code wrote outside the
  stdout redirect):
      "<<<SANDBOX>>>{...}\n"

The only output the host should interpret is the single prefixed line. Any
other stdout written by the kernel loop itself (e.g. stray prints before
redirect takes effect) is ignored.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import signal
import sys
import traceback
from typing import Any

_MARKER = "<<<SANDBOX>>>"


def _run_cell(code: str, globals_: dict[str, Any]) -> tuple[str, str, str, bool]:
    """Exec all statements; if the last is an expression, eval it and include
    its repr as the cell result. Returns (stdout, stderr, result_repr, failed).
    """
    out_buf, err_buf = io.StringIO(), io.StringIO()
    result_repr = ""
    failed = False

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        err_buf.write(traceback.format_exc())
        return "", err_buf.getvalue(), "", True

    last_expr = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = ast.Expression(body=tree.body[-1].value)
        tree = ast.Module(body=tree.body[:-1], type_ignores=[])

    with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
        try:
            if tree.body:
                exec(compile(tree, "<cell>", "exec"), globals_)
            if last_expr is not None:
                value = eval(compile(last_expr, "<cell>", "eval"), globals_)
                if value is not None:
                    try:
                        result_repr = repr(value)
                    except Exception:
                        result_repr = f"<unreprable {type(value).__name__}>"
        except TimeoutError as e:
            err_buf.write(f"TimeoutError: {e}\n")
            failed = True
        except SystemExit as e:
            err_buf.write(f"SystemExit suppressed: {e}\n")
            failed = True
        except BaseException:
            err_buf.write(traceback.format_exc())
            failed = True

    return out_buf.getvalue(), err_buf.getvalue(), result_repr, failed


def _timeout_handler(signum, frame):
    raise TimeoutError("execution exceeded the per-cell timeout")


def main() -> None:
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    globals_: dict[str, Any] = {"__name__": "__main__", "__builtins__": __builtins__}

    # Send a ready marker so the host knows the jail is up.
    sys.stdout.write(f"{_MARKER}{json.dumps({'ready': True})}\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stdout.write(
                f"{_MARKER}{json.dumps({'id': None, 'error': f'bad request: {e}'})}\n"
            )
            sys.stdout.flush()
            continue

        call_id = req.get("id")
        code = req.get("code", "")
        timeout = int(req.get("timeout", 30))

        signal.alarm(timeout)
        try:
            stdout, stderr, result, failed = _run_cell(code, globals_)
        finally:
            signal.alarm(0)

        resp = {
            "id": call_id,
            "stdout": stdout,
            "stderr": stderr,
            "result": result,
            "failed": failed,
        }
        sys.stdout.write(f"{_MARKER}{json.dumps(resp)}\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
