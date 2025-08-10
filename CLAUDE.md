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
- `GET /chat/stream-sse` - Primary streaming endpoint with session management
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