# Installation Guide

This guide covers installation on **Linux**, **macOS**, and **Windows**. Read the platform-specific notes carefully — some features have different behaviour or availability depending on your OS.

---

## Platform Support Summary

| Feature | Linux | macOS | Windows (native) | Windows (WSL2) |
|---|---|---|---|---|
| Chat / streaming | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Local Ollama models | ✅ | ✅ | ✅ | ✅ |
| Cloud models (Groq, OpenAI, Anthropic) | ✅ | ✅ | ✅ | ✅ |
| PDF upload (text extraction) | ✅ | ✅ | ✅ | ✅ |
| PDF upload (page images / vision) | ✅ | ✅ | ✅ with installer | ✅ |
| Python code execution sandbox | ✅ Full | ✅ rlimit mode | ❌ Not supported | ✅ Full |
| `bwrap` full sandboxing | ✅ optional | ❌ N/A | ❌ N/A | ✅ optional |
| `start.sh` one-command launch | ✅ | ✅ | ❌ Use manual steps | ✅ |

> **Windows native caveat:** `sandbox.py` imports the POSIX-only `resource` module at the top level. This causes an `ImportError` on native Windows, making the backend unable to start. **Windows users must either use WSL2 (recommended) or disable code execution** (see the Windows section for details).

---

## Prerequisites (all platforms)

- **Python 3.11 or newer** — [python.org/downloads](https://www.python.org/downloads/)
- **Ollama** — [ollama.com](https://ollama.com/) (available for Linux, macOS, and Windows)
- **Git** — to clone the repository

Optional (only needed for specific features):
- A **Tavily API key** for web search — [app.tavily.com](https://app.tavily.com/)
- API keys for cloud models: Groq, OpenAI, Anthropic, OpenRouter

---

## Linux

### 1 — System dependencies

**Ubuntu / Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git poppler-utils

# Optional: bubblewrap for full Python sandbox isolation
sudo apt install bubblewrap
```

**Fedora / RHEL / CentOS:**
```bash
sudo dnf install python3 python3-pip git poppler-utils

# Optional: bubblewrap
sudo dnf install bubblewrap
```

**Arch Linux:**
```bash
sudo pacman -S python python-pip git poppler

# Optional: bubblewrap
sudo pacman -S bubblewrap
```

> **`poppler-utils`** is required for the PDF-to-image feature (multimodal vision with PDFs). Without it, PDF text extraction still works, but page images will not be sent to vision-capable models.

> **`bubblewrap`** (`bwrap`) provides full filesystem + network isolation for sandboxed Python execution. If it is not installed, the sandbox automatically falls back to resource-limits-only mode (RLIMIT_AS / CPU / NPROC). On Ubuntu 24.04+ with AppArmor restricting unprivileged user namespaces, bwrap may be installed but blocked — the code detects this at runtime and falls back silently.

### 2 — Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve          # runs in background; skip if already running
```

### 3 — Clone the repository

```bash
git clone https://github.com/nitishkthakur/OfflineLM.git
cd OfflineLM
```

### 4 — Create a Python virtual environment and install dependencies

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 5 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` in any editor and fill in the keys you need. Only `TAVILY_API_KEY` is required if you want web search; all LLM API keys are optional — leave any you do not use blank.

```
OLLAMA_BASE_URL=http://localhost:11434   # change only if Ollama runs elsewhere
TAVILY_API_KEY=tvly-...
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### 6 — Pull an Ollama model

```bash
ollama pull qwen3:4b          # fast 4B model, good default
ollama pull gemma3:12b        # larger, higher quality
ollama pull qwen2.5vl:7b      # vision-capable (needed for PDF image mode)
```

### 7 — Start the application

**One command (recommended):**
```bash
chmod +x start.sh
./start.sh
```

This starts the backend on port 8000 and the frontend on port 3000, then prints the URLs.

**Manually (two terminals):**
```bash
# Terminal 1 — backend
cd backend
source .venv/bin/activate
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 — frontend
cd frontend
python3 -m http.server 3000
```

Open **http://localhost:3000** in your browser.

---

## macOS

### 1 — Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2 — System dependencies

```bash
brew install python@3.12 git poppler
```

> **`poppler`** provides the `pdftoppm` binary used by `pdf2image` to convert PDF pages to images. Without it, PDF text extraction still works but page images won't be passed to vision models.

> **`bwrap` (bubblewrap) is not available on macOS.** The Python code execution sandbox automatically detects this and switches to resource-limits-only (`rlimit_only`) mode. Code execution still works; the trade-off is that there is no filesystem or network isolation between the sandbox process and the host. Consider setting `"enabled": false` in the `code_execution` block of `config.json` if this is a concern.

### 3 — Install Ollama

Download and install from [ollama.com/download](https://ollama.com/download), or via Homebrew:
```bash
brew install ollama
ollama serve          # start in a terminal, or use the menu-bar app
```

### 4 — Clone the repository

```bash
git clone https://github.com/nitishkthakur/OfflineLM.git
cd OfflineLM
```

### 5 — Create a virtual environment and install dependencies

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 6 — Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your keys (see the Linux section above for the full list).

### 7 — Pull an Ollama model

```bash
ollama pull qwen3:4b
ollama pull qwen2.5vl:7b     # vision-capable
```

### 8 — Start the application

**One command:**
```bash
chmod +x start.sh
./start.sh
```

**Manually (two terminals):**
```bash
# Terminal 1 — backend
cd backend && source .venv/bin/activate
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 — frontend
cd frontend && python3 -m http.server 3000
```

Open **http://localhost:3000** in your browser.

---

## Windows

### Option A — WSL2 (Strongly recommended)

WSL2 gives you a full Linux environment inside Windows. All features work identically to the Linux installation above, including the sandbox and `bwrap`. This is the recommended path.

#### 1 — Enable WSL2

Open **PowerShell as Administrator** and run:
```powershell
wsl --install
```
Restart your machine if prompted. By default this installs **Ubuntu**. You can also choose a different distro:
```powershell
wsl --install -d Ubuntu-24.04
```

#### 2 — Open a WSL terminal

Launch "Ubuntu" (or your chosen distro) from the Start menu. You now have a Linux shell. Follow the **Linux installation steps** above from within this shell.

The project files can live either inside WSL (e.g. `~/OfflineLM`) or on your Windows drive (accessible at `/mnt/c/Users/<YourName>/...`). Storing them inside WSL gives better I/O performance.

#### 3 — Access Ollama

If you install the **Windows Ollama app** (recommended), it already listens on `http://localhost:11434` and is reachable from WSL2. No extra configuration is needed.

Alternatively, install Ollama inside WSL itself:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

#### 4 — Open the frontend in your Windows browser

Once the servers are running inside WSL, open your Windows browser at:
```
http://localhost:3000
```
WSL2 automatically port-forwards to the Windows host.

---

### Option B — Native Windows (limited — code execution disabled)

> **Important:** `sandbox.py` imports the POSIX-only `resource` module at the module level. On native Windows Python, this causes an `ImportError` at startup. You **must** disable the code execution feature in `config.json` before starting the backend.

#### 1 — Install prerequisites

- **Python 3.11+** from [python.org](https://www.python.org/downloads/) — check "Add Python to PATH" during install.
- **Git** from [git-scm.com](https://git-scm.com/download/win).
- **Ollama** for Windows from [ollama.com/download](https://ollama.com/download).
- **Poppler for Windows** (for PDF image support):
  1. Download the latest release from [github.com/oschwartz10612/poppler-windows/releases](https://github.com/oschwartz10612/poppler-windows/releases).
  2. Extract it (e.g. to `C:\poppler`).
  3. Add `C:\poppler\Library\bin` to your system `PATH`.

#### 2 — Disable code execution

Open `config.json` and set `"enabled": false` inside the `code_execution` block:
```json
"code_execution": {
    "enabled": false,
    ...
}
```
This prevents `sandbox.py` from being initialised, avoiding the `ImportError` from the `resource` module. All other features (chat, PDF, web search, artifacts) remain fully functional.

#### 3 — Clone the repository

Open **Command Prompt** or **Git Bash**:
```cmd
git clone https://github.com/nitishkthakur/OfflineLM.git
cd OfflineLM
```

#### 4 — Create a virtual environment and install dependencies

```cmd
python -m venv backend\.venv
backend\.venv\Scripts\activate
pip install --upgrade pip
pip install -r backend\requirements.txt
```

#### 5 — Configure environment variables

```cmd
copy .env.example .env
```

Edit `.env` with Notepad or any editor.

#### 6 — Pull an Ollama model

Open a new Command Prompt (the Ollama Windows app starts the server automatically):
```cmd
ollama pull qwen3:4b
```

#### 7 — Start the application manually

`start.sh` is a Bash script and does not run natively on Windows. Start both servers manually.

**Terminal 1 — backend:**
```cmd
cd backend
.venv\Scripts\activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — frontend:**
```cmd
cd frontend
python -m http.server 3000
```

Open **http://localhost:3000** in your browser.

> If you have **Git Bash** installed, you can run `./start.sh` from a Git Bash terminal (it sources `bin/activate` which exists on POSIX paths). However, the virtual environment activated via Git Bash's Python is different from the one in `cmd.exe`. Use one consistently.

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | No | Ollama server URL. Default: `http://localhost:11434` |
| `TAVILY_API_KEY` | For web search | Get from [app.tavily.com](https://app.tavily.com/) |
| `GROQ_API_KEY` | For Groq models | Get from [console.groq.com](https://console.groq.com/) |
| `OPENAI_API_KEY` | For OpenAI models | Get from [platform.openai.com](https://platform.openai.com/) |
| `ANTHROPIC_API_KEY` | For Claude models | Get from [console.anthropic.com](https://console.anthropic.com/) |
| `OPENROUTER_API_KEY` | For OpenRouter models | Get from [openrouter.ai](https://openrouter.ai/) |

---

## Configuration (`config.json`)

Key settings you may want to change:

| Key | Default | Description |
|---|---|---|
| `default_model` | `"minimax-m2.7:cloud"` | Ollama model ID pre-selected in the UI |
| `default_backend` | `"react_agent"` | Agent backend: `"react_agent"` or `"deep_agent"` |
| `ollama_base_url` | `"http://localhost:11434"` | Can also be overridden by `OLLAMA_BASE_URL` env var |
| `code_execution.enabled` | `true` | Set to `false` on Windows native or to disable sandboxing |
| `code_execution.timeout_seconds` | `30` | Max seconds a single code cell may run |
| `ui_defaults.num_ctx` | `100000` | Default context window size sent to Ollama |

---

## Sandbox Behaviour by Platform

The Python code execution sandbox (`python_execute` tool) behaves differently depending on what is available at runtime:

| Mode | Trigger condition | Isolation |
|---|---|---|
| `bwrap` (full) | Linux + `bubblewrap` installed + kernel allows user namespaces | Filesystem, network, PID, IPC, UTS namespaces isolated |
| `rlimit_only` (fallback) | Linux without working `bwrap`, or macOS | CPU, memory, process count, file size limited via `setrlimit` — no filesystem isolation |
| Disabled | `"enabled": false` in `config.json`, or Windows native | No code execution; tool returns an error |

The sandbox mode is detected automatically at startup; no configuration is needed on Linux or macOS.

---

## Verifying the Installation

Once both servers are running, open your browser at **http://localhost:3000** and:

1. The model dropdown in the sidebar should list your pulled Ollama models.
2. Send a test message — you should see a streamed response.
3. Click the web search toggle and ask a current-events question (requires `TAVILY_API_KEY`).
4. Upload a small PDF — text extraction should succeed on all platforms; page images require `poppler`.
5. Ask the agent to `run some Python` — verify the trace sidebar shows `tool_start` / `tool_end` events (Linux / macOS only; disabled on Windows native).

---

## Troubleshooting

**Backend won't start on Windows:**
Most likely the `resource` module import error. Ensure `"code_execution": {"enabled": false}` is set in `config.json`.

**`pdf2image` raises `PDFInfoNotInstalledError` or similar:**
`poppler` is not installed or not on your `PATH`. Install it per the platform steps above and restart the backend.

**Ollama connection refused:**
Make sure `ollama serve` is running. Check the URL in `.env` (`OLLAMA_BASE_URL`) and `config.json` (`ollama_base_url`) are consistent. On WSL2, use `http://localhost:11434` — WSL2 routes this to the Windows Ollama process automatically.

**Port already in use:**
Another process is using port 8000 or 3000. Either stop it or edit `start.sh` (and the frontend's `fetch` base URL in `app.js`) to use different ports.

**`bwrap` installed but sandbox still shows `rlimit_only` on Linux:**
Your kernel's AppArmor policy is blocking unprivileged user namespaces. This is common on Ubuntu 24.04+. The fallback is automatic. To check: `bwrap --unshare-user /bin/true && echo ok`.

**Model dropdown is empty:**
The backend cannot reach Ollama. Verify `ollama serve` is running and the URL is correct. Also confirm you have pulled at least one model: `ollama list`.
