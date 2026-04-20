"""Shared tools available to all agent backends."""
import ast
import json
import math
import operator
import os
import re
from pathlib import Path
from typing import Literal, Optional

from tavily import TavilyClient


SYSTEM_PROMPT = """You are Jarvis, a highly capable AI assistant with access to various tools.

You can:
1. Search the web for current information using the internet_search tool.
2. Save artifacts (markdown files, code, notes) to disk using save_artifact.
3. List and read previously saved artifacts.
4. Use calculate to evaluate any mathematical expression precisely — always prefer
   this over mental arithmetic to avoid rounding errors.
5. Use pct_change to compute percentage change between two values.
6. Execute Python code via python_execute for data analysis, spreadsheet crunching,
   and programmatic verification. The kernel persists variables across calls.
7. Spawn subagents when needed for specialized tasks or parallel execution.
8. If the user asks you to be careful or critique your task, use a subagent to give
   targeted feedback on how to improve your work, then improve based on that feedback.

IMPORTANT — use calculate and pct_change liberally:
- Any time you compute a number (even "simple" ones), call calculate instead of
  doing it yourself. LLMs make arithmetic mistakes; the tool does not.
- Any time you compare two figures and want a growth or change percentage, call
  pct_change instead of computing it inline.

When the user asks you to create or save something, use the save_artifact tool.
When searching for information, use the internet_search tool.
When analysing a spreadsheet's data programmatically, prefer python_execute with
pandas over reproducing the whole table inline.

## PDF Document Context

When a user attaches a PDF, its extracted text is wrapped in a tag of the form:

  <pdf-filename-FILENAME>
  ... document text ...
  </pdf-filename-FILENAME>

where FILENAME is the actual name of the PDF file (e.g. <pdf-filename-report.pdf>).
Use this tag to know exactly which document a piece of text came from.
If multiple PDFs are attached across the conversation, each has its own tag with a
distinct filename — refer to documents by their filename when answering.

## Spreadsheet (Excel / CSV) Context

When a user attaches an Excel or CSV file, the data is wrapped in a tag of the form:

  <excel-filename-FILENAME>
  <sheet name="SheetName">
  {"columns":["col1","col2",...],"rows":[[val1,val2,...],[val3,val4,...],...]}
  </sheet>
  </excel-filename-FILENAME>

- "columns" lists the header names once; "rows" contains the data as a JSON array
  of arrays where each inner array corresponds to one row in column order.
- Numeric values are rounded to integers.
- Multi-sheet workbooks have one <sheet> block per sheet, all inside the same outer tag.
- Always use the calculate and pct_change tools when performing arithmetic on
  spreadsheet values — never compute figures inline.

Be helpful, accurate, and thorough in your responses."""


# ── Safe AST-based expression evaluator ───────────────────────────────────────
#
# Only a whitelist of AST node types is allowed.  No exec, no imports,
# no attribute access, no comprehensions — purely numeric computation.

_BINARY_OPS: dict = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
}

_UNARY_OPS: dict = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_FUNCS: dict = {
    "abs":    abs,
    "round":  round,
    "sum":    sum,
    "min":    min,
    "max":    max,
    "pow":    pow,
    "sqrt":   math.sqrt,
    "floor":  math.floor,
    "ceil":   math.ceil,
    "log":    math.log,       # log(x) = ln(x);  log(x, base) = log_base(x)
    "log10":  math.log10,
    "exp":    math.exp,
}

_ALLOWED_NAMES: dict = {
    "pi":  math.pi,
    "e":   math.e,
}


def _safe_eval(node: ast.AST):
    """Recursively evaluate a whitelisted AST node and return a numeric result."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Non-numeric constant {node.value!r} is not allowed.")

    if isinstance(node, ast.BinOp):
        op_fn = _BINARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Operator '{type(node.op).__name__}' is not supported.")
        left  = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("Division by zero.")
        if isinstance(node.op, ast.Pow) and abs(right) > 300:
            raise ValueError(
                "Exponent magnitude > 300 is rejected to prevent runaway computation. "
                "Use log/exp for very large powers."
            )
        return op_fn(left, right)

    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unary operator '{type(node.op).__name__}' is not supported.")
        return op_fn(_safe_eval(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError(
                "Only plain function calls are allowed (e.g. sqrt(x)), "
                "not method calls (e.g. obj.method(x))."
            )
        fn_name = node.func.id
        fn = _ALLOWED_FUNCS.get(fn_name)
        if fn is None:
            raise ValueError(
                f"Function '{fn_name}' is not allowed. "
                f"Allowed: {', '.join(sorted(_ALLOWED_FUNCS))}."
            )
        args = [_safe_eval(a) for a in node.args]
        return fn(*args)

    if isinstance(node, ast.Name):
        val = _ALLOWED_NAMES.get(node.id)
        if val is None:
            raise ValueError(
                f"Name '{node.id}' is not recognised. "
                f"Allowed names: {', '.join(sorted(_ALLOWED_NAMES))}."
            )
        return val

    if isinstance(node, (ast.List, ast.Tuple)):
        return [_safe_eval(e) for e in node.elts]

    raise ValueError(
        f"Expression construct '{type(node).__name__}' is not supported. "
        "Only arithmetic operations, whitelisted functions, and numeric literals are allowed."
    )


def _fmt_number(value) -> str:
    """Format a numeric result cleanly for display."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN (result is undefined)"
        if math.isinf(value):
            return "Infinity (result overflowed)"
        # Whole-number float → show as integer with thousands separator
        if value == math.floor(value) and abs(value) < 1e15:
            return f"{int(value):,}"
        # General float: up to 10 significant figures, trailing zeros stripped
        return f"{value:.10g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


# ── Public tool functions ──────────────────────────────────────────────────────

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression precisely and return the result as a string.

    Use this tool for every arithmetic computation, no matter how simple. Language
    models make arithmetic mistakes; this tool does not. Never compute numbers inline
    when this tool is available.

    Supported operators: + (add), - (subtract), * (multiply), / (divide),
    // (floor divide), % (modulo), ** (exponentiation), () (grouping).

    Supported functions: abs(x), round(x, n), sum([...]), min(...), max(...),
    pow(x, y), sqrt(x), floor(x), ceil(x), log(x), log(x, base), log10(x), exp(x).

    Supported constants: pi, e.

    Not supported: variable assignment (x = 5), named variables in expressions,
    imports, string operations, list comprehensions, or function definitions.
    All named values must be substituted with their actual numbers before calling
    this tool. Exponents above 300 in magnitude are rejected to prevent runaway
    computation — use log/exp for very large powers instead.

    Division by zero returns an error message, not infinity.

    Examples:
      "(2_450_000 - 1_830_000) / 2_450_000 * 100"  →  "25.30612245"   (gross margin %)
      "round(850_000 / 120_000, 2)"                 →  "7.08"          (coverage ratio)
      "1_000_000 * (1.08 ** 5)"                     →  "1,469,328"     (compound growth)
      "sum([12.5, 8.3, 15.1]) / 3"                  →  "11.96666667"   (weighted avg)
      "1,500,000 * 1.08"                             →  "1,620,000"     (comma notation)

    Args:
        expression: The mathematical expression to evaluate as a plain string.
            May contain numeric literals (integers or floats), arithmetic operators,
            parentheses, any of the supported function calls, and the constants pi
            and e. Comma thousand-separators within numeric literals are stripped
            automatically (e.g. "1,500,000" is treated as 1500000); commas that
            separate function arguments are preserved. Underscores in numeric
            literals also work natively (e.g. 1_500_000). No variable names —
            substitute all values with their actual numbers before calling.
    """
    # Strip only numeric thousand-separator commas (digit,digit) so that
    # "1,500,000" → "1500000" but "round(3.14, 2)" stays intact.
    cleaned = re.sub(r'(?<=\d),(?=\d)', '', expression).strip()
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        return f"Syntax error in expression: {exc}"
    try:
        result = _safe_eval(tree)
    except ValueError as exc:
        return f"Evaluation error: {exc}"
    except Exception as exc:
        return f"Unexpected error: {exc}"
    return _fmt_number(result)


def pct_change(old_value: float, new_value: float, round_to: int = 2) -> str:
    """Compute the percentage change from a base value to a new value.

    Formula: ((new_value − old_value) / old_value) × 100.

    Returns a labelled string that is ready to embed in a financial narrative,
    e.g. "+18.18% increase" or "-17.65% decrease". Always use this tool instead
    of computing percentage changes inline.

    Use calculate("new - old") when you need the absolute difference rather than
    the relative percentage change. Do not use this tool when old_value is zero —
    percentage change is mathematically undefined for a zero base.

    Note on negative base values: when old_value is negative (e.g. a prior-year
    operating loss), the sign of the result follows standard formula convention
    and may appear counter-intuitive — always add a brief explanation in context.

    Examples:
      pct_change(44_000_000, 52_000_000)          →  "+18.18% increase"
      pct_change(5_100_000, 4_200_000)            →  "-17.65% decrease"
      pct_change(1_000_000, 1_000_000)            →  "0.00% (no change)"
      pct_change(300_000, 450_000, round_to=1)    →  "+50.0% increase"

    Args:
        old_value: The base (starting) value to measure change from, e.g. prior-year
            revenue or cost. Must be non-zero — returns an error if zero.
        new_value: The comparison (ending) value to measure change to, e.g.
            current-year revenue or cost.
        round_to: Number of decimal places in the returned percentage string.
            Defaults to 2. Use 0 for whole-number output (e.g. "+18% increase"),
            or 3–4 for high-precision financial reporting.
    """
    if old_value == 0:
        return (
            "Error: old_value is zero — percentage change is undefined (division by zero). "
            "Use calculate to find the absolute difference instead."
        )
    change  = ((new_value - old_value) / old_value) * 100
    rounded = round(change, round_to)
    fmt     = f"{rounded:.{round_to}f}"
    if rounded == 0:
        return f"0.{'0' * round_to}% (no change)"
    sign      = "+" if rounded > 0 else ""
    direction = "increase" if rounded > 0 else "decrease"
    return f"{sign}{fmt}% {direction}"


# ── Factory ────────────────────────────────────────────────────────────────────

def _get_tavily_client() -> Optional[TavilyClient]:
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key:
        return TavilyClient(api_key=api_key)
    return None


def make_tools(
    artifacts_dir: Path,
    web_search_enabled: bool = True,
    *,
    conversation_id: Optional[str] = None,
    sandbox_registry=None,
    allow_network: bool = False,
    code_exec_config: Optional[dict] = None,
) -> list:
    """Return a list of LangChain-compatible tool functions for the given request context.

    ``python_execute`` is registered only when ``code_exec_config['enabled']`` is
    True, a ``sandbox_registry`` is provided, and a ``conversation_id`` is given.
    """

    def internet_search(
        query: str,
        max_results: int = 10,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
    ) -> str:
        """Search the public web via the Tavily search API and return ranked results.

        ## When to use
        - The user asks a question whose answer changes over time (news, prices,
          rates, recent releases, current leadership, live sports scores).
        - You need facts that post-date your training cutoff, or the user
          explicitly asks you to "search", "look up", or "find the latest".
        - A claim in the conversation needs independent verification against a
          primary source.

        ## When NOT to use
        - The answer is already present in the conversation, an attached PDF, an
          attached spreadsheet, or a saved artifact — read those first.
        - The question is conceptual, definitional, or mathematical — you already
          know the answer or can compute it with calculate.
        - The user has disabled web search — the tool will return a disabled
          notice and will not make a network call.

        ## Writing a good query
        - Include the specific proper nouns, figures, and time window that matter:
          "Apple Q4 2023 services revenue year-over-year", not "apple earnings".
        - For a recurring figure, include the year or fiscal period explicitly.
        - Do not paste the entire user question — extract the searchable nouns.

        ## Returns
        A JSON string with shape ``{"results": [{"title", "url", "content", ...}]}``.
        ``content`` is a short extracted snippet; use ``include_raw_content=true``
        only when the snippet is too sparse to answer the question.

        Args:
            query: A focused search query. Prefer 4–10 content-rich words; avoid
                filler ("what is", "please tell me"). Include dates or proper
                nouns that narrow the result set.
            max_results: Number of results to retrieve (1–20). Defaults to 10.
                Use 3–5 for a targeted fact lookup where you want to cite one
                source; use 10–20 for exploratory research where you will
                synthesize across sources.
            topic: Corpus to search. ``"general"`` (default) for broad web
                results, ``"news"`` for recent event coverage and headlines
                (past days/weeks), ``"finance"`` for company financials,
                earnings, market data, SEC filings.
            include_raw_content: When True, each result also carries the full
                parsed page text. Defaults to False because raw content is
                lengthy. Turn on only after an initial search where the snippets
                were insufficient.
        """
        if not web_search_enabled:
            return "Web search is currently disabled."
        client = _get_tavily_client()
        if not client:
            return "Web search is not available. TAVILY_API_KEY not configured."
        try:
            results = client.search(
                query,
                max_results=max_results,
                include_raw_content=include_raw_content,
                topic=topic,
            )
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Search error: {str(e)}"

    def save_artifact(filename: str, content: str) -> str:
        """Write text content to a named file in the project's artifacts directory.

        ## When to use
        - The user explicitly asks you to "save", "write", "create a file",
          "export", or "put this in a document".
        - You have produced a substantial deliverable (analysis, report, code
          module, long plan) that the user will want to keep or download.
        - You need to persist intermediate output so a later tool call — for
          instance python_execute or a subagent — can read it back from disk.

        ## When NOT to use
        - For short conversational replies — just return the text. Saving every
          message clutters the artifacts list.
        - To write binary data. Only plain text (UTF-8) is supported.

        ## Behaviour
        - Overwrites silently if ``filename`` already exists. To preserve an
          older version, pick a distinct name.
        - Files written here are visible to list_artifacts/read_artifact, are
          surfaced in the UI's "View Artifacts" modal, and are the same
          ``/workspace`` directory python_execute sees — so a file saved here
          can be loaded in Python as ``open('/workspace/<filename>')``.

        Args:
            filename: Plain filename with extension, e.g. ``"analysis.md"``,
                ``"forecast_2025.csv"``, ``"notes.txt"``. Must not contain path
                separators (``/``, ``\\``) or ``..`` — everything is written to
                the fixed artifacts directory.
            content: The complete file body as a string. Pass the finished
                content; this tool does not append. Include trailing newline if
                POSIX conventions matter.
        """
        filepath = artifacts_dir / filename
        with open(filepath, "w") as f:
            f.write(content)
        return f"Artifact saved to {filepath}"

    def list_artifacts() -> str:
        """List the filenames of every file currently saved in the artifacts directory.

        ## When to use
        - Before calling read_artifact, to confirm the exact filename you want.
        - Before calling save_artifact, to avoid accidentally overwriting a file
          when the user might want both versions.
        - When the user asks "what have we saved?" or "what files do I have?".

        ## Returns
        Newline-separated filenames (one per line), sorted as the filesystem
        returns them. If nothing is saved, returns the literal string
        ``"No artifacts found."`` — check for that before trying to parse.

        This tool takes no arguments.
        """
        files = list(artifacts_dir.glob("*"))
        if not files:
            return "No artifacts found."
        return "\n".join([f.name for f in files if f.is_file()])

    def read_artifact(filename: str) -> str:
        """Read and return the full UTF-8 text content of a saved artifact file.

        ## When to use
        - You need to continue editing or reference a document you (or an
          earlier turn) saved with save_artifact.
        - The user refers to a file by name ("update the analysis I saved
          earlier", "what did we write in forecast.md?").
        - A multi-step workflow needs to resume from persisted state.

        ## When NOT to use
        - If the content is already visible in the current conversation —
          re-reading wastes tokens.
        - For spreadsheet or PDF contents — those arrive via the upload
          pipeline and are already embedded in the conversation; you do not
          need to re-read them from disk.

        ## Returns
        The raw file contents as a string. If the file does not exist, returns
        a literal ``"Artifact <filename> not found."`` — call list_artifacts to
        see the exact names.

        Args:
            filename: Exact filename, including extension, matching an entry
                returned by list_artifacts (case sensitive). Do not include any
                directory path.
        """
        filepath = artifacts_dir / filename
        if not filepath.exists():
            return f"Artifact {filename} not found."
        with open(filepath, "r") as f:
            return f.read()

    tools_list = [calculate, pct_change, internet_search, save_artifact, list_artifacts, read_artifact]

    # ── Python code execution (optional, sandboxed) ───────────────────────────
    if (
        code_exec_config
        and code_exec_config.get("enabled")
        and sandbox_registry is not None
        and conversation_id
    ):
        default_timeout = int(code_exec_config.get("timeout_seconds", 30))
        net_note = (
            "Network access is ENABLED — you may use urllib/requests to fetch URLs."
            if allow_network
            else "Network access is DISABLED — any socket call will fail with a network error. "
                 "Do not attempt HTTP requests, pip installs, or DNS lookups."
        )

        def python_execute(code: str, timeout: int = default_timeout) -> str:
            sess = sandbox_registry.get_or_create(conversation_id, allow_network)
            resp = sess.execute(code, timeout=timeout)
            parts = []
            if resp.get("stdout"):
                parts.append(f"stdout:\n{resp['stdout']}")
            if resp.get("stderr"):
                parts.append(f"stderr:\n{resp['stderr']}")
            if resp.get("result"):
                parts.append(f"result: {resp['result']}")
            if resp.get("failed") and not resp.get("stderr"):
                parts.append("execution failed (no stderr captured)")
            if not parts:
                parts.append("(no output)")
            return "\n\n".join(parts)

        python_execute.__doc__ = f"""Execute Python code inside a persistent, sandboxed kernel and return its output.

        ## Execution model
        - The kernel is a **real Python process** that lives for the entire
          conversation. Variables, imports, and function definitions from
          earlier calls are still defined in later calls — just like a Jupyter
          notebook. You do **not** need to re-import libraries or redefine
          helpers on every call.
        - State is reset only when the sandbox restarts: the
          "Sandbox Network" toggle flipped, idle-kill timeout elapsed, or a
          previous cell wedged past its SIGALRM and was force-killed. Files
          you wrote to ``/workspace`` always survive a restart.

        ## Environment
        - Working directory: ``/workspace`` (same directory as save_artifact /
          read_artifact). Read spreadsheets the user attached by loading them
          from this path, e.g. ``pd.read_excel('/workspace/<filename>.xlsx')``.
        - Preinstalled packages (use **these** rather than anything else —
          ``pip install`` is NOT available at runtime, and extra dependencies
          will fail with ModuleNotFoundError):
            * ``pandas``      — dataframes, spreadsheet I/O, group-by, rolling
            * ``numpy``       — arrays, vectorised math, ``polyfit``, ``linalg``
            * ``scipy``       — ``scipy.stats`` (``linregress``, ``ttest_ind``,
                                ``pearsonr``), ``scipy.optimize``, ``scipy.signal``
            * ``scikit-learn``— ``LinearRegression``, ``LogisticRegression``,
                                ``RandomForest*``, ``train_test_split``,
                                ``StandardScaler``, ``KMeans``, metrics
            * ``openpyxl``    — backend for ``pd.read_excel`` / ``to_excel``
            * ``matplotlib``  — plotting; save figures to ``/workspace/*.png``
                                with ``fig.savefig(...)`` (no display server —
                                always save, never ``plt.show()``)
            * ``requests``    — HTTP client (only when network is enabled)
          The full Python standard library is also available.
        - Filesystem: ``/workspace`` is writable; everything else is read-only
          or a tmpfs — writes to ``/etc``, ``/usr``, ``$HOME`` will fail.
        - Network: {net_note}

        ## Preferred package choices (do not reach for alternatives)
        - Linear regression → ``scipy.stats.linregress`` for a quick
          slope+intercept+r+pvalue, **or** ``numpy.polyfit(x, y, 1)``, **or**
          ``sklearn.linear_model.LinearRegression`` for the scikit-learn API.
          Do not attempt ``statsmodels`` (not installed).
        - Classification / clustering / feature preprocessing → scikit-learn.
          Do not attempt ``xgboost``, ``lightgbm``, ``catboost``, ``torch``,
          ``tensorflow`` — none are available.
        - Optimisation / curve fitting → ``scipy.optimize.minimize`` or
          ``scipy.optimize.curve_fit``.
        - Statistical tests / distributions → ``scipy.stats``.
        - Plotting → ``matplotlib`` only (no ``seaborn``, no ``plotly``).
        - HTTP fetches → ``requests`` only (no ``httpx``, no ``aiohttp``).
        - Excel read/write → ``pandas`` + the installed ``openpyxl`` engine
          (no ``xlrd``, no ``xlsxwriter``).

        ## When to use
        - Parsing or aggregating an attached spreadsheet with pandas (group-bys,
          pivots, filters, column math) instead of eyeballing the raw rows.
        - Verifying a multi-step computation programmatically — write the steps
          as code and let Python do the arithmetic.
        - Producing a derived dataset the user wants saved (e.g. write a CSV
          summary into ``/workspace``, then tell the user it is saved).
        - Running a quick simulation, regex pass, or date-math calculation
          that would be error-prone by hand.

        ## When NOT to use
        - Single arithmetic expressions — use calculate; it is faster and its
          output is structured for a model to read.
        - Percentage-change comparisons — use pct_change.
        - Fetching live web data — use internet_search. ``python_execute``'s
          network access is off by default and, even when enabled, is not a
          search engine.
        - Writing deliverables the user wants to keep — use save_artifact so
          the file shows up in the Artifacts UI and can be downloaded.

        ## Output
        The tool returns a single string concatenating up to three labelled
        sections separated by blank lines:
          - ``stdout:\\n<captured prints>``
          - ``stderr:\\n<captured tracebacks>``
          - ``result: <repr of the final expression>``
        Sections are omitted when empty. If the final statement is an
        expression (Jupyter-style), its ``repr()`` appears in ``result``.
        Assignments, loops, and ``print(...)`` statements do not populate
        ``result`` — use ``print`` or end the cell with a bare expression.

        ## Examples
        Inspect a spreadsheet::
            import pandas as pd
            df = pd.read_excel('/workspace/statement.xlsx')
            df.head()                  # ends in expression → shows in 'result'

        Compute then save::
            summary = df.groupby('category')['amount'].sum()
            summary.to_csv('/workspace/category_totals.csv')
            print(summary.to_string())

        Continue in the next call — ``df`` and ``summary`` are still defined::
            df['amount'].quantile([0.25, 0.5, 0.75])

        Linear regression with scipy::
            import numpy as np
            from scipy.stats import linregress
            x = np.arange(len(df))
            y = df['close'].to_numpy()
            r = linregress(x, y)
            print(f"slope={{r.slope:.4f}} r2={{r.rvalue**2:.4f}}")

        Same thing with scikit-learn (when you want the full sklearn API)::
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(x.reshape(-1, 1), y)
            print("slope=", model.coef_[0], "intercept=", model.intercept_)

        ## Failure modes
        - Syntax errors and uncaught exceptions appear in ``stderr`` with a
          full traceback; the kernel stays alive for the next call.
        - Timeouts: the cell is interrupted by SIGALRM after ``timeout``
          seconds; if the cell ignores the signal (rare — C extensions in
          tight loops), the kernel is killed and restarted, and all in-memory
          state is lost. Files under ``/workspace`` are preserved.
        - Output larger than the configured limit is truncated with a
          ``[… N chars truncated …]`` note so you can retry with smaller
          print statements.

        Args:
            code: Python source to execute. May contain any number of
                statements (imports, assignments, loops, function/class defs).
                Make the final line a bare expression if you want its value
                echoed back. Do not wrap code in ```fences``` — pass raw source.
            timeout: Hard wall-clock limit in seconds ({default_timeout}s
                default from config). Keep it tight for quick probes; raise
                for data-loading cells that legitimately need more time.
        """

        tools_list.append(python_execute)

    return tools_list
