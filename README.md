# OfflineLM (Jarvis)

A full-featured AI agent chat application supporting local Ollama models alongside cloud providers (Groq, OpenAI, Anthropic). Features a ChatGPT-style interface with streaming responses, deep agent reasoning, sandboxed Python code execution, web search, and PDF support.

## Features

- **Local-first**: Run entirely offline with [Ollama](https://ollama.com) models
- **Cloud providers**: Groq, OpenAI, Anthropic (optional — set API keys in `.env`)
- **Deep Agent mode**: Multi-step reasoning with researcher + critic subagents
- **Sandboxed Python execution**: Persistent Jupyter-style kernel with numpy, pandas, scipy, scikit-learn, matplotlib, requests
- **Web search**: Tavily-powered search with configurable result count
- **PDF support**: Upload PDFs — text extraction + vision multimodal (for capable models)
- **Streaming responses** with real-time thought trace sidebar
- **Conversation export** as PDF

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally (`ollama serve`)
- `poppler-utils` — required for PDF-to-image conversion:
  ```bash
  # Ubuntu/Debian
  sudo apt install poppler-utils

  # macOS
  brew install poppler
  ```

## Quick Start

### 1. Clone and set up the backend

```bash
git clone https://github.com/nitishkthakur/OfflineLM.git
cd OfflineLM

python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env — only add keys for providers you want to use
```

Key variables (all optional except for cloud model usage):

| Variable | Purpose |
|---|---|
| `OLLAMA_BASE_URL` | Ollama endpoint (default: `http://localhost:11434`) |
| `GROQ_API_KEY` | Groq cloud models |
| `OPENAI_API_KEY` | OpenAI GPT models |
| `ANTHROPIC_API_KEY` | Anthropic Claude models |
| `TAVILY_API_KEY` | Web search (required for search toggle) |

### 3. Start the app

**Option A — one command (recommended):**

```bash
chmod +x start.sh
./start.sh
```

This starts the backend on `:8000` and frontend on `:3000`, then opens your browser automatically.

**Option B — manually:**

```bash
# Terminal 1 — backend
cd backend
source .venv/bin/activate
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 — frontend
cd frontend
python3 -m http.server 3000
```

Then open **http://localhost:3000** in your browser.

### 4. Pull a model in Ollama

```bash
ollama pull qwen3.5:4b       # default model (fast, 4B params)
ollama pull gemma4:26b       # larger, more capable
ollama pull qwen2-vl:7b      # vision-capable model
```

## Configuration

`config.json` controls defaults:

```json
{
  "default_model": "qwen3.5:4b",
  "default_backend": "react_agent",
  "ollama_base_url": "http://localhost:11434",
  "code_execution": {
    "enabled": true,
    "timeout_seconds": 30,
    "packages": ["pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "requests", "openpyxl"]
  }
}
```

## Project Structure

```
OfflineLM/
├── backend/
│   ├── main.py              # FastAPI app + API routes
│   ├── agents/
│   │   ├── react_agent.py   # Default single-agent mode
│   │   └── deep_agent.py    # Multi-step deep research agent
│   ├── sandbox.py           # Python execution sandbox manager
│   ├── sandbox_runner.py    # In-jail kernel loop
│   ├── tools.py             # Tool definitions (search, code exec, artifacts)
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── config.json              # App configuration
├── .env.example             # Environment variable template
└── start.sh                 # One-command startup script
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/models` | List available Ollama models |
| `POST` | `/chat` | Send a message (SSE streaming) |
| `POST` | `/upload` | Upload a PDF |
| `GET` | `/model-info/{model_id}` | Model capabilities (vision, tools, thinking) |
| `GET` | `/artifacts` | List saved artifacts |
| `GET` | `/artifacts/{filename}` | Get artifact content |
| `POST` | `/download-pdf` | Export conversation as PDF |

## License

MIT
