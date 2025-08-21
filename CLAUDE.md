# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OfflineLM is a streaming chat application that provides an interactive interface for conversing with Ollama models. The project implements both basic agent functionality and advanced streaming chat capabilities with PDF export features.

## Development Commands

### Starting the Application
- **Windows**: `start.bat` - Automated launcher that handles environment setup, dependency installation, server startup, and browser opening
- **Manual FastAPI**: `python fastapi_streaming_improved.py` - Starts the FastAPI server on port 8001
- **Basic Agent**: `python basic_ollama_agent_with_post.py` - Runs example usage of the OllamaAgent and OllamaChat classes

### Dependency Management
- **Install dependencies**: `pip install -r requirements_streaming.txt`
- **Virtual environment setup**: `python -m venv venv` then `venv/Scripts/activate` (Windows) or `source venv/bin/activate` (Linux/Mac)

### Testing Ollama Connection
- **Check Ollama status**: `curl http://localhost:11434/api/tags`
- **Start Ollama service**: `ollama serve`

## Architecture

### Core Components

1. **OllamaAgent** (`basic_ollama_agent_with_post.py:8-248`)
   - Tool-based agent with function calling capabilities
   - Supports structured output via Pydantic schemas
   - Implements `invoke()` and `invoke_plus_next_call()` methods for single and two-step processes

2. **OllamaChat** (`basic_ollama_agent_with_post.py:249-453`)
   - Conversation-focused chat interface with history management
   - Provides both synchronous (`chat()`) and streaming (`chat_stream()`, `chat_stream_async()`) methods
   - Maintains conversation context across multiple exchanges

3. **FastAPI Server** (`fastapi_streaming_improved.py`)
   - Full-featured web server with streaming chat endpoints
   - Server-Sent Events (SSE) implementation for real-time streaming
   - Model management, PDF export, and conversation history features

### Key Features

- **Streaming Implementation**: Real-time word-by-word response generation using SSE
- **Model Management**: Dynamic model switching with automatic resource cleanup
- **Think Tag Processing**: Special handling of `<think>` tags for reasoning display
- **PDF Export**: Conversation export with markdown formatting and syntax highlighting
- **Message Formatting**: Backend markdown processing with code block support

### API Endpoints

- `GET /` - Serves the main chat interface
- `GET /chat/stream-sse` - Legacy streaming endpoint with URL parameters
- `POST /chat/stream-sse-v2` - New streaming endpoint with configuration support
- `POST /change-model` - Model switching with conversation history preservation
- `GET /models` - Lists available Ollama models
- `POST /download-chat-pdf` - Generates PDF export of conversations
- `POST /stop-stream/{session_id}` - Interrupts active streaming sessions

## Configuration

### Default Models
- Primary: Auto-detected first available model from Ollama
- Fallback: `qwen2.5:7b`
- Alternative models: `gemma3:4b-it-fp16`, `llama3.2:3b`, `phi3:mini`

### Server Settings
- **Port**: 8001 (FastAPI server)
- **Ollama Endpoint**: `http://localhost:11434/api/chat`
- **Keepalive**: 30 minutes for loaded models
- **Static Files**: Served from current directory

### Dependencies
- FastAPI, Uvicorn (web framework)
- aiohttp (async HTTP client)
- ReportLab (PDF generation)
- markdown, pygments (content formatting)
- Pydantic (data validation)

## Chat Configuration System

### Overview
The application now supports a comprehensive configuration system that allows dynamic control of various features through a structured config object passed from frontend to backend.

### Configuration Structure
```python
class ChatConfig(BaseModel):
    search: SearchConfig = SearchConfig()
    rag: RagConfig = RagConfig()
    # Future feature flags for extensibility
    summary_enabled: bool = False
    voice_enabled: bool = False
    analytics_enabled: bool = False

class SearchConfig(BaseModel):
    enabled: bool = False
    count: int = Field(default=5, ge=1, le=20)

class RagConfig(BaseModel):
    enabled: bool = False
    chunks: int = Field(default=5, ge=1, le=10)
    text: Optional[str] = None
```

### Feature Configurations

#### Search Configuration
- **enabled**: Boolean flag to enable/disable web search functionality
- **count**: Number of search results to retrieve (1-20)
- Requires `TAVILY_API_KEY` environment variable
- When enabled, searches the web and provides context to the LLM

#### RAG Configuration
- **enabled**: Boolean flag to enable/disable RAG (Retrieval-Augmented Generation)
- **chunks**: Number of relevant chunks to retrieve (1-10)
- **text**: Optional text content for RAG processing
- Currently implemented as placeholder (returns empty context)
- Future implementation will support document chunking and similarity search

#### Future Feature Flags
- **summary_enabled**: Enable automatic conversation summarization
- **voice_enabled**: Enable voice input/output capabilities
- **analytics_enabled**: Enable usage analytics and tracking

### API Usage

#### New Endpoint (Recommended)
```http
POST /chat/stream-sse-v2
Content-Type: application/json

{
  "message": "Your question here",
  "config": {
    "search": {
      "enabled": true,
      "count": 5
    },
    "rag": {
      "enabled": false,
      "chunks": 5
    },
    "summary_enabled": false,
    "voice_enabled": false,
    "analytics_enabled": false
  }
}
```

#### Legacy Endpoint (Backward Compatible)
```http
GET /chat/stream-sse?message=query&search_enabled=true&search_count=5&rag_enabled=false&rag_chunks=5
```

### Frontend Integration
The frontend automatically reads UI toggle states and constructs the configuration object:
- Search toggle and count input → `config.search`
- RAG toggle and chunks input → `config.rag`
- Future UI elements will map to respective config flags

### Processing Flow
1. **Frontend**: User presses send button
2. **Frontend**: Reads UI state and constructs config object
3. **Frontend**: Sends message + config to backend via POST
4. **Backend**: Parses config and enables requested features
5. **Backend**: Processes search/RAG if enabled
6. **Backend**: Streams response with enhanced context

## Development Notes

### Stream Management
- Active streams are tracked in `active_streams` dictionary
- Session-based interruption support via `/stop-stream/{session_id}`
- Automatic OLLAMA process management to prevent resource conflicts

### Content Processing
- Think tags are processed server-side with special formatting
- Markdown rendering includes syntax highlighting support
- PDF export preserves code blocks and formatting

### Model Switching
- Automatic stopping of inactive models to conserve resources
- Conversation history preservation across model changes
- Graceful fallback handling for model loading failures

## File Structure

### Core Files
- `basic_ollama_agent_with_post.py` - Agent and chat classes
- `fastapi_streaming_improved.py` - Web server implementation
- `requirements_streaming.txt` - Python dependencies
- `start.bat` - Windows launcher script

### Frontend Assets
- `minimal_ui_streaming.html` - Chat interface
- `styles.css` - UI styling
- `script.js` - Client-side JavaScript

### Documentation
- `STREAMING_README.md` - Detailed usage instructions and examples
- `docs/` - Reference documents for the knowledge base

### Legacy/Examples
- `Older_versions/` - Previous implementations and notebooks
- `templates/` - Alternative UI templates