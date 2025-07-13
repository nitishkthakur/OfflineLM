#!/usr/bin/env python3
"""
Improved FastAPI streaming chat with proper Server-Sent Events implementation.
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import json
import uuid
from basic_ollama_agent_with_post import OllamaChat
from typing import List, Dict

app = FastAPI(title="Ollama Streaming Chat API - Improved")

# Initialize the chat agent
chat_agent = OllamaChat(model="qwen2.5:7b")

class ChatMessage(BaseModel):
    message: str

class ModelChangeRequest(BaseModel):
    model: str
    conversation_history: List[Dict[str, str]] = []

class ModelInfo(BaseModel):
    name: str
    size: str = "Unknown"
    modified_at: str = "Unknown"

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the minimal UI frontend HTML template with streaming support."""
    try:
        with open("/home/nitish/Documents/github/PublicReportResearch/minimal_ui_streaming.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: minimal_ui_streaming.html not found</h1>", status_code=404)

@app.get("/models")
async def get_available_models():
    """Get list of available Ollama models."""
    try:
        # Try to get models from Ollama API
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get('models', []):
                models.append({
                    'name': model.get('name', 'Unknown'),
                    'size': model.get('size', 'Unknown'),
                    'modified_at': model.get('modified_at', 'Unknown')
                })
            return {"models": models}
    except Exception as e:
        print(f"Error fetching models from Ollama: {e}")
    
    # Fallback to common models if API fails
    fallback_models = [
        {"name": "qwen2.5:7b", "size": "4.7GB", "modified_at": "Recently"},
        {"name": "gemma3:4b-it-fp16", "size": "2.4GB", "modified_at": "Recently"},
        {"name": "llama3.2:3b", "size": "2.0GB", "modified_at": "Recently"},
        {"name": "phi3:mini", "size": "2.3GB", "modified_at": "Recently"},
    ]
    return {"models": fallback_models}

@app.post("/change-model")
async def change_model(request: ModelChangeRequest):
    """Change the active model and optionally restore conversation history."""
    try:
        global chat_agent
        
        # Create new chat agent with the selected model
        chat_agent = OllamaChat(model=request.model)
        print("\n\n Model Requested: {}".format(request.model))
        # Restore conversation history if provided
        if request.conversation_history:
            print("Here")
            chat_agent.conversation_history = request.conversation_history
            print("Am I hgere?")
        print(f"Model changed to {chat_agent.model} with history restored: {len(request.conversation_history)} messages")
        
        return {
            "status": "success", 
            "message": f"Model changed to {request.model}",
            "history_restored": len(request.conversation_history) > 0
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/chat/stream-sse")
async def chat_stream_sse(message: str, model: str = None):
    """Stream chat responses using Server-Sent Events with EventSource."""
    def generate():
        try:
            # Use specified model if provided
            current_agent = chat_agent
            if model and model != chat_agent.model:
                # Create temporary agent with specified model but preserve history
                temp_agent = OllamaChat(model=model)
                temp_agent.conversation_history = chat_agent.conversation_history.copy()
                current_agent = temp_agent
            
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting response...'})}\n\n"
            
            for chunk in current_agent.chat_stream(message):
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            
            # Update global agent's history if we used a different model
            if model and model != chat_agent.model:
                chat_agent.conversation_history = current_agent.conversation_history.copy()
            
            yield f"data: {json.dumps({'type': 'done', 'message': 'Response complete'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

@app.post("/chat/stream")
async def chat_stream_post(chat_message: ChatMessage):
    """Stream chat responses using POST request (alternative method)."""
    def generate():
        try:
            for chunk in chat_agent.chat_stream(chat_message.message):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield f"data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

@app.post("/chat")
async def chat_regular(chat_message: ChatMessage):
    """Regular (non-streaming) chat endpoint."""
    try:
        response = chat_agent.chat(chat_message.message)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@app.post("/clear")
async def clear_history():
    """Clear the chat history."""
    chat_agent.clear_history()
    return {"status": "cleared"}

@app.get("/history")
async def get_history():
    """Get the current chat history."""
    return {"history": chat_agent.get_history()}

@app.get("/messages")
async def get_messages():
    """Get the current conversation messages formatted for the frontend."""
    try:
        messages = []
        for msg in chat_agent.get_history():
            messages.append({
                'type': msg['role'],
                'content': msg['content']
            })
        return {"messages": messages}
    except Exception as e:
        return {"messages": [], "error": str(e)}

@app.post("/chat/stream-minimal")
async def chat_stream_minimal(chat_message: ChatMessage):
    """Stream chat responses for the minimal UI using Server-Sent Events."""
    def generate():
        try:
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting response...'})}\n\n"
            
            full_response = ""
            for chunk in chat_agent.chat_stream(chat_message.message):
                full_response += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done', 'message': 'Response complete', 'full_response': full_response})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

if __name__ == "__main__":
    import uvicorn
    print("Starting Improved FastAPI streaming chat server...")
    print("Open http://localhost:8001 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8001)
