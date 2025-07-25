# Ollama Streaming Chat Implementation

This implementation adds streaming functionality to the OllamaChat class, allowing you to see words as they are generated by the LLM instead of waiting for the complete response.

## Features

- **Streaming Chat**: Real-time word-by-word response generation
- **Conversation History**: Maintains context across multiple exchanges
- **Async Support**: Compatible with modern async web frameworks
- **Multiple UI Options**: Command-line, Flask, and FastAPI examples

## Usage

### Basic Streaming Chat

```python
from basic_ollama_agent_with_post import OllamaChat

# Initialize chat agent
chat_agent = OllamaChat(model="qwen2.5:7b")

# Stream response word by word
for chunk in chat_agent.chat_stream("What is artificial intelligence?"):
    print(chunk, end="", flush=True)
```

### Command Line Interface

```bash
python streaming_chat_ui.py
```

### Web Interface (FastAPI)

```bash
# Install dependencies
pip install -r requirements_streaming.txt

# Start the server
python fastapi_streaming_chat.py

# Open http://localhost:8000 in your browser
```

### Web Interface (Flask)

```bash
python streaming_chat_ui.py web
```

## Methods Available

### OllamaChat Class

- `chat(prompt)` - Regular chat with full response
- `chat_stream(prompt)` - Streaming chat with word-by-word generation
- `chat_stream_async(prompt)` - Async version for web frameworks
- `clear_history()` - Clear conversation history
- `get_history()` - Get current conversation history

### Key Parameters

- `model`: The Ollama model to use (default: "gemma3:4b-it-fp16")
- `endpoint`: Ollama API endpoint (default: "http://localhost:11434/api/chat")
- `stream`: Set to True for streaming responses

## Implementation Details

The streaming functionality works by:

1. Setting `stream: true` in the Ollama API request
2. Processing the response line by line as it arrives
3. Yielding each content chunk as it's received
4. Maintaining conversation history for context

## Benefits

- **Improved User Experience**: Users see responses being generated in real-time
- **Faster Perceived Response Time**: Users start reading before the full response is complete
- **Better Engagement**: More interactive and dynamic conversation flow
- **Reduced Waiting Time**: Especially beneficial for longer responses

## Requirements

- Python 3.7+
- Ollama running locally or accessible endpoint
- For web examples: FastAPI/Flask and related dependencies
