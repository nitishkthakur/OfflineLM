#!/usr/bin/env python3
"""
Improved FastAPI streaming chat with proper Server-Sent Events implementation.
"""

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import uuid
import os
import tempfile
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, blue, darkgray
import markdown
from basic_ollama_agent_with_post import OllamaChat
from typing import List, Dict, Optional
import threading
import time
import subprocess
import re

# Try to import pygments for syntax highlighting, fall back gracefully
try:
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name, TextLexer
    from pygments.formatters import HtmlFormatter
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False
    print("Warning: Pygments not available, syntax highlighting disabled")

app = FastAPI(title="Ollama Streaming Chat API - Improved")

# Mount static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="."), name="static")

# Global variables for model management
default_model = None
chat_agent = None

# Track active streaming sessions
active_streams = {}
stream_locks = {}

# Message formatting functions (migrated from frontend)
def process_think_tags(content: str) -> str:
    """
    Process think tags by converting them to styled HTML format.
    Migrated from frontend JavaScript to backend Python.
    """
    # Check if content starts with <think> and contains </think>
    think_regex = r'^(<think>)([\s\S]*?)(<\/think>)([\s\S]*)$'
    match = re.match(think_regex, content)
    
    if match:
        open_tag, think_content, close_tag, remaining_content = match.groups()
        
        # Clean up the think content - preserve newlines and structure
        clean_think_content = think_content.strip()
        
        # Format the think section with special styling
        formatted_think_section = f'<div class="think-section"><em>{open_tag}\n{clean_think_content}\n{close_tag}</em></div>'
        
        # Add proper spacing after think section
        if remaining_content.strip():
            return formatted_think_section + '\n\n' + remaining_content.strip()
        else:
            return formatted_think_section
    
    # If we're in the middle of streaming and see <think> at the start but no closing tag yet,
    # we'll format it differently to show it's in progress
    if content.startswith('<think>') and '</think>' not in content:
        # Format as in-progress think section
        return f'<div class="think-section think-streaming"><em>{content}</em></div>'
    
    return content

def highlight_code_block(code: str, language: str = '') -> str:
    """
    Highlight code block using Pygments if available, otherwise return plain HTML.
    """
    if not PYGMENTS_AVAILABLE:
        return f'<pre><code class="language-{language}">{code}</code></pre>'
    
    try:
        if language:
            lexer = get_lexer_by_name(language, stripall=True)
        else:
            lexer = TextLexer()
        
        formatter = HtmlFormatter(
            style='github-dark',
            noclasses=False,
            cssclass='highlight'
        )
        
        highlighted = highlight(code, lexer, formatter)
        return highlighted
        
    except Exception as e:
        # Fallback to plain code block if highlighting fails
        print(f"Code highlighting error: {e}")
        return f'<pre><code class="language-{language}">{code}</code></pre>'

def format_message(content: str) -> str:
    """
    Format message content with markdown processing, LaTeX math, and syntax highlighting.
    Migrated from frontend JavaScript to backend Python.
    """
    try:
        # Process think tags first before markdown parsing
        processed_content = process_think_tags(content)
        
        # Configure markdown with extensions including LaTeX math support
        extensions = [
            'codehilite',
            'fenced_code', 
            'tables',
            'toc',
            'nl2br'
        ]
        
        extension_configs = {
            'codehilite': {
                'css_class': 'highlight',
                'use_pygments': PYGMENTS_AVAILABLE
            }
        }
        
        # Try to add LaTeX math support if available
        try:
            import mdx_math
            extensions.append('mdx_math')
            extension_configs['mdx_math'] = {
                'enable_dollar_delimiter': True,  # Enable $...$ for inline math
                'add_preview': False  # Don't add preview functionality
            }
        except ImportError:
            print("Warning: mdx_math not available, LaTeX math expressions won't be processed on backend")
            # LaTeX will still work via frontend KaTeX rendering
        
        md = markdown.Markdown(
            extensions=extensions,
            extension_configs=extension_configs
        )
        
        # Parse markdown
        html = md.convert(processed_content)
        
        return html
        
    except Exception as error:
        print(f"Error parsing markdown: {error}")
        # Fallback to simple HTML conversion
        return process_think_tags(content).replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;').replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')

def clean_content_for_backend(content: str) -> str:
    """
    Clean and sanitize content for backend processing.
    Migrated from frontend JavaScript to backend Python.
    """
    # For now, return content as-is but this could include:
    # - HTML sanitization
    # - Content validation  
    # - Security checks
    return content

def calculate_chat_stats() -> Dict:
    """
    Calculate chat statistics on the backend.
    """
    try:
        message_count = len(chat_agent.get_history()) if chat_agent else 0
        current_model = chat_agent.model if chat_agent else default_model
        
        return {
            'messageCount': message_count,
            'currentModel': current_model,
            'status': 'Ready'
        }
    except Exception as e:
        print(f"Error calculating stats: {e}")
        return {
            'messageCount': 0,
            'currentModel': default_model or 'Unknown',
            'status': 'Error'
        }

def get_first_available_model():
    """Get the first available model from OLLAMA server."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            if models:
                return models[0].get('name', 'qwen2.5:7b')
    except Exception as e:
        print(f"Could not fetch models from OLLAMA: {e}")
    
    # Fallback to a common model
    return "qwen2.5:7b"

def initialize_chat_agent():
    """Initialize the chat agent with the first available model."""
    global default_model, chat_agent
    default_model = get_first_available_model()
    chat_agent = OllamaChat(model=default_model)
    print(f"Initialized chat agent with model: {default_model}")
    return default_model

# Initialize the chat agent with the first available model
default_model = initialize_chat_agent()

class ChatMessage(BaseModel):
    message: str

class ModelChangeRequest(BaseModel):
    model: str
    conversation_history: List[Dict[str, str]] = []

class ModelStatusUpdate(BaseModel):
    status: str
    message: str
    timestamp: str

class ModelInfo(BaseModel):
    name: str
    size: str = "Unknown"
    modified_at: str = "Unknown"

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the minimal UI frontend HTML template with streaming support."""
    try:
        with open("minimal_ui_streaming.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: minimal_ui_streaming.html not found</h1>", status_code=404)

@app.get("/models")
async def get_available_models():
    """Get list of available Ollama models with consistent default."""
    try:
        # Try to get models from Ollama API
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get('models', []):
                models.append({
                    'name': model.get('name', 'Unknown'),
                    'size': model.get('size', 'Unknown'),
                    'modified_at': model.get('modified_at', 'Unknown')
                })
            
            # Include the default model info
            result = {
                "models": models,
                "default_model": default_model if models else None,
                "status": "success"
            }
            return result
    except Exception as e:
        print(f"Error fetching models from Ollama: {e}")
    
    # Fallback to common models if API fails, with consistent default
    fallback_models = [
        {"name": default_model, "size": "Unknown", "modified_at": "Recently"},
        {"name": "qwen2.5:7b", "size": "4.7GB", "modified_at": "Recently"},
        {"name": "gemma3:4b-it-fp16", "size": "2.4GB", "modified_at": "Recently"},
        {"name": "llama3.2:3b", "size": "2.0GB", "modified_at": "Recently"},
        {"name": "phi3:mini", "size": "2.3GB", "modified_at": "Recently"},
    ]
    return {
        "models": fallback_models,
        "default_model": default_model,
        "status": "fallback"
    }

async def stop_all_other_models(current_model: str) -> List[str]:
    """Stop all OLLAMA models except the current one."""
    stopped_models = []
    try:
        import requests
        
        # First, get list of running models
        ps_response = requests.get("http://localhost:11434/api/ps")
        if ps_response.status_code == 200:
            running_models = ps_response.json().get('models', [])
            
            for model_info in running_models:
                model_name = model_info.get('name', '')
                if model_name and model_name != current_model:
                    try:
                        # Stop the model
                        stop_response = requests.post(
                            "http://localhost:11434/api/generate",
                            json={
                                "model": model_name,
                                "keep_alive": 0  # This stops the model immediately
                            }
                        )
                        if stop_response.status_code == 200:
                            stopped_models.append(model_name)
                            print(f"Successfully stopped model: {model_name}")
                    except Exception as e:
                        print(f"Error stopping model {model_name}: {e}")
        
    except Exception as e:
        print(f"Error in stop_all_other_models: {e}")
    
    return stopped_models

async def ensure_model_loaded_with_keepalive(model_name: str) -> bool:
    """Ensure a model is loaded with 30 minute keepalive."""
    try:
        import requests
        
        # Send a simple generation request with keepalive to ensure model is loaded
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": "",  # Empty prompt to just load the model
                "stream": False,
                "keep_alive": "30m"  # Set 30 minute keepalive
            },
            timeout=60  # Give it time to load if needed
        )
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error ensuring model {model_name} is loaded: {e}")
        return False

@app.post("/change-model")
async def change_model(request: ModelChangeRequest):
    """Change the active model, stop others, and set keepalive to 30 minutes."""
    try:
        global chat_agent
        
        print(f"\n\n Model Requested: {request.model}")
        
        # Step 1: Stop all other models
        stopped_models = await stop_all_other_models(request.model)
        
        # Step 2: Ensure the new model is loaded with keepalive
        model_loaded = await ensure_model_loaded_with_keepalive(request.model)
        
        if not model_loaded:
            return {
                "status": "error", 
                "message": f"Failed to load model {request.model}",
                "stopped_models": stopped_models
            }
        
        # Step 3: Create new chat agent with the selected model
        chat_agent = OllamaChat(model=request.model)
        
        # Step 4: Restore conversation history if provided
        if request.conversation_history:
            chat_agent.conversation_history = request.conversation_history
        
        print(f"Model changed to {chat_agent.model} with history restored: {len(request.conversation_history)} messages")
        print(f"Stopped models: {stopped_models}")
        
        return {
            "status": "success", 
            "message": f"Model changed to {request.model}",
            "history_restored": len(request.conversation_history) > 0,
            "stopped_models": stopped_models,
            "keepalive_set": "30m"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/model-status")
async def get_model_status():
    """Get status of currently running OLLAMA models."""
    try:
        import requests
        
        response = requests.get("http://localhost:11434/api/ps")
        if response.status_code == 200:
            data = response.json()
            running_models = data.get('models', [])
            
            model_status = []
            for model in running_models:
                model_status.append({
                    'name': model.get('name', 'Unknown'),
                    'size': model.get('size', 'Unknown'),
                    'size_vram': model.get('size_vram', 'Unknown'),
                    'until': model.get('until', 'Unknown'),
                    'status': 'running'
                })
            
            return {
                "status": "success",
                "models": model_status,
                "current_model": chat_agent.model if chat_agent else None,
                "default_model": default_model
            }
        else:
            return {
                "status": "error",
                "message": "Failed to get model status from OLLAMA",
                "models": []
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "models": [],
            "default_model": default_model
        }

@app.get("/default-model")
async def get_default_model():
    """Get the current default model."""
    return {
        "default_model": default_model,
        "current_model": chat_agent.model if chat_agent else None,
        "status": "success"
    }

@app.get("/chat/stream-sse")
async def chat_stream_sse(message: str, model: str = None, session: str = None):
    """Stream chat responses using Server-Sent Events with EventSource."""
    def generate():
        try:
            # Register this streaming session
            if session:
                active_streams[session] = True
                stream_locks[session] = threading.Lock()
            
            # Use specified model if provided
            current_agent = chat_agent
            if model and model != chat_agent.model:
                # Create temporary agent with specified model but preserve history
                temp_agent = OllamaChat(model=model)
                temp_agent.conversation_history = chat_agent.conversation_history.copy()
                current_agent = temp_agent
            
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting response...'})}\n\n"
            
            # Stream with interruption checking and backend formatting
            full_response = ""
            try:
                for chunk in current_agent.chat_stream(message):
                    # Check if stream should be stopped
                    if session and not active_streams.get(session, False):
                        print(f"Stream {session} was stopped by user")
                        # Send stop signal to OLLAMA (sync version inside generator)
                        import asyncio
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                # Schedule the stop function to run
                                loop.create_task(force_stop_ollama_generation(current_agent.model))
                            else:
                                asyncio.run(force_stop_ollama_generation(current_agent.model))
                        except Exception as e:
                            print(f"Error scheduling stop: {e}")
                        yield f"data: {json.dumps({'type': 'stopped', 'message': 'Stream stopped by user'})}\n\n"
                        break
                    
                    full_response += chunk
                    
                    # Format the accumulated response using backend formatting
                    try:
                        formatted_content = format_message(full_response)
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk, 'formatted_content': formatted_content, 'raw_content': full_response})}\n\n"
                    except Exception as format_error:
                        print(f"Formatting error: {format_error}")
                        # Fallback to sending raw chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk, 'raw_content': full_response})}\n\n"
                    
                    # Small delay to allow interruption checking
                    time.sleep(0.01)
                else:
                    # Stream completed normally - send final formatted content
                    if session and active_streams.get(session, False):
                        try:
                            final_formatted = format_message(full_response)
                            stats = calculate_chat_stats()
                            yield f"data: {json.dumps({'type': 'done', 'message': 'Response complete', 'final_formatted': final_formatted, 'stats': stats})}\n\n"
                        except Exception as final_format_error:
                            print(f"Final formatting error: {final_format_error}")
                            yield f"data: {json.dumps({'type': 'done', 'message': 'Response complete'})}\n\n"
            except Exception as stream_error:
                print(f"Error in streaming: {stream_error}")
                if session:
                    try:
                        import asyncio
                        loop = asyncio.get_event_loop()
                        loop.create_task(force_stop_ollama_generation(current_agent.model))
                    except Exception as e:
                        print(f"Error scheduling stop: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(stream_error)})}\n\n"
            
            # Update global agent's history if we used a different model and stream wasn't stopped
            if model and model != chat_agent.model and full_response:
                chat_agent.conversation_history = current_agent.conversation_history.copy()
            
        except Exception as e:
            print(f"Error in chat_stream_sse: {e}")
            if session:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    loop.create_task(force_stop_ollama_generation(model or chat_agent.model))
                except Exception as stop_e:
                    print(f"Error scheduling stop: {stop_e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            # Clean up session
            if session:
                active_streams.pop(session, None)
                stream_locks.pop(session, None)
    
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

async def force_stop_ollama_generation(model_name: str):
    """Force stop OLLAMA model generation to reduce CPU usage to zero."""
    try:
        import requests
        
        print(f"Force stopping OLLAMA generation for model: {model_name}")
        
        # Method 1: Send empty generation request with immediate stop
        try:
            stop_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": 0  # Stop immediately
                },
                timeout=2
            )
            print(f"Stop request sent: {stop_response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Stop request failed: {e}")
        
        # Method 2: Try to interrupt any running OLLAMA processes
        try:
            # Send SIGTERM to OLLAMA processes (graceful stop)
            result = subprocess.run(
                ["pkill", "-TERM", "-f", "ollama"],
                capture_output=True,
                timeout=2
            )
            if result.returncode == 0:
                print("Sent SIGTERM to OLLAMA processes")
                time.sleep(0.5)  # Give time for graceful shutdown
            
            # If still running, send SIGKILL (force stop)
            result2 = subprocess.run(
                ["pgrep", "-f", "ollama"],
                capture_output=True,
                timeout=1
            )
            if result2.returncode == 0 and result2.stdout.strip():
                print("OLLAMA processes still running, sending SIGKILL")
                subprocess.run(
                    ["pkill", "-KILL", "-f", "ollama"],
                    capture_output=True,
                    timeout=2
                )
                time.sleep(0.2)
                
                # Restart OLLAMA service
                print("Restarting OLLAMA service...")
                subprocess.Popen(
                    ["nohup", "ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
                
        except Exception as proc_error:
            print(f"Error with process management: {proc_error}")
        
    except Exception as e:
        print(f"Error in force_stop_ollama_generation: {e}")

@app.post("/stop-stream/{session_id}")
async def stop_stream(session_id: str):
    """Stop an active streaming session."""
    try:
        if session_id in active_streams:
            print(f"Stopping stream session: {session_id}")
            active_streams[session_id] = False
            
            # Force stop OLLAMA generation
            await force_stop_ollama_generation(chat_agent.model)
            
            return {
                "status": "success",
                "message": f"Stream {session_id} stopped",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": f"Stream session {session_id} not found",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        print(f"Error stopping stream {session_id}: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

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
            raw_content = msg['content']
            formatted_content = format_message(raw_content)
            
            messages.append({
                'type': msg['role'],
                'content': formatted_content,
                'raw_content': raw_content
            })
        
        stats = calculate_chat_stats()
        return {"messages": messages, "stats": stats}
    except Exception as e:
        return {"messages": [], "error": str(e), "stats": calculate_chat_stats()}

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

def process_latex_for_pdf(content):
    """
    Process LaTeX expressions for PDF generation by converting them to readable text.
    Since ReportLab doesn't support LaTeX rendering, we'll convert common expressions.
    """
    import re
    
    # Convert display math $$...$$ to text representation
    content = re.sub(r'\$\$(.*?)\$\$', r'\n[MATH: \1]\n', content, flags=re.DOTALL)
    
    # Convert inline math $...$ to text representation
    content = re.sub(r'\$([^$]+)\$', r'[MATH: \1]', content)
    
    # Convert LaTeX brackets \[...\] to text representation
    content = re.sub(r'\\?\[(.*?)\\?\]', r'\n[MATH: \1]\n', content, flags=re.DOTALL)
    
    # Convert LaTeX parentheses \(...\) to text representation
    content = re.sub(r'\\?\((.*?)\\?\)', r'[MATH: \1]', content)
    
    return content

def process_think_tags_for_pdf(content):
    """
    Process think tags for PDF generation by converting them to readable format.
    Removes HTML formatting and converts to markdown-friendly format.
    """
    import re
    
    # First process LaTeX expressions
    content = process_latex_for_pdf(content)
    
    # Check if content starts with <think> and contains </think>
    think_regex = r'^(<think>)([\s\S]*?)(<\/think>)([\s\S]*)$'
    match = re.match(think_regex, content)
    
    if match:
        open_tag, think_content, close_tag, remaining_content = match.groups()
        
        # Format think section for PDF
        think_section = f"**💭 Reasoning Process:**\n\n_{think_content.strip()}_\n\n---\n\n"
        
        # Return formatted content
        if remaining_content.strip():
            return think_section + remaining_content.strip()
        else:
            return think_section
    
    # Remove any HTML div tags that might have been passed from frontend
    content = re.sub(r'<div[^>]*class="think-section[^"]*"[^>]*>', '**💭 Reasoning Process:**\n\n', content)
    content = re.sub(r'<em>(.*?)</em>', r'_\1_', content, flags=re.DOTALL)
    content = re.sub(r'</div>', '\n\n---\n\n', content)
    
    return content

@app.post("/download-chat-pdf")
async def download_chat_pdf(background_tasks: BackgroundTasks):
    """
    Asynchronously generates a PDF export of the current chat conversation, preserving markdown formatting and code blocks, and returns it as a downloadable file response.
    Args:
        background_tasks (BackgroundTasks): FastAPI background tasks manager for scheduling file cleanup after download.
    Returns:
        FileResponse: A FastAPI response object for downloading the generated PDF file.
    Raises:
        HTTPException: If there are no messages to export or if PDF generation fails.
    Process Overview:
        - Retrieves chat history from the chat agent.
        - Formats messages with markdown and code block support for PDF rendering.
        - Generates a styled PDF file with metadata and message formatting.
        - Schedules temporary file cleanup after download.
        - Returns the PDF as a downloadable file response.
    """
    """Generate and download chat conversation as PDF with markdown formatting."""
    try:
        # Get conversation messages
        messages = []
        for msg in chat_agent.get_history():
            messages.append({
                'type': msg['role'],
                'content': msg['content']
            })
        
        if not messages:
            raise HTTPException(status_code=400, detail="No messages to export")
        
        # Create temporary file for PDF
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"chat-export-{timestamp}.pdf"
        filepath = os.path.join(temp_dir, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=1*inch
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            textColor=black,
            alignment=1  # Center alignment
        )
        
        metadata_style = ParagraphStyle(
            'MetadataStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=darkgray,
            spaceAfter=6
        )
        
        user_header_style = ParagraphStyle(
            'UserHeader',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=blue,
            spaceBefore=12,
            spaceAfter=6,
            leftIndent=0
        )
        
        assistant_header_style = ParagraphStyle(
            'AssistantHeader',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=darkgray,
            spaceBefore=12,
            spaceAfter=6,
            leftIndent=0
        )
        
        content_style = ParagraphStyle(
            'ContentStyle',
            parent=styles['Normal'],
            fontSize=10,
            leftIndent=20,
            spaceAfter=8,
            spaceBefore=4,
            leading=14,
            allowWidows=0,
            allowOrphans=0
        )
        
        code_style = ParagraphStyle(
            'CodeStyle',
            parent=styles['Code'],
            fontSize=9,
            leftIndent=30,
            spaceAfter=8,
            spaceBefore=4,
            backgroundColor='#f5f5f5',
            fontName='Courier',
            borderWidth=1,
            borderColor=darkgray,
            borderPadding=6
        )
        
        # Build PDF content
        story = []
        
        # Title
        story.append(Paragraph("Chat Conversation Export", title_style))
        story.append(Spacer(1, 12))
        
        # Metadata
        now = datetime.now()
        story.append(Paragraph(f"<b>Exported:</b> {now.strftime('%Y-%m-%d %H:%M:%S')}", metadata_style))
        story.append(Paragraph(f"<b>Model:</b> {chat_agent.model}", metadata_style))
        story.append(Paragraph(f"<b>Total Messages:</b> {len(messages)}", metadata_style))
        story.append(Spacer(1, 20))
        
        # Process each message
        for i, message in enumerate(messages):
            # Message header
            if message['type'] == 'user':
                story.append(Paragraph("👤 <b>User:</b>", user_header_style))
            else:
                story.append(Paragraph("🤖 <b>Assistant:</b>", assistant_header_style))
            
            # Process content - preserve markdown formatting and order
            content = message['content']
            
            try:
                # Handle think tags specially for PDF generation
                content = process_think_tags_for_pdf(content)
                
                # Handle content with code blocks while preserving order
                if '```' in content:
                    parts = content.split('```')
                    
                    for j, part in enumerate(parts):
                        if j % 2 == 0:  # Regular text (not code)
                            if part.strip():
                                # Process markdown for regular text
                                html_content = markdown.markdown(part.strip())
                                # Clean up HTML tags for ReportLab
                                html_content = html_content.replace('<p>', '').replace('</p>', '<br/><br/>')
                                html_content = html_content.replace('<strong>', '<b>').replace('</strong>', '</b>')
                                html_content = html_content.replace('<em>', '<i>').replace('</em>', '</i>')
                                html_content = html_content.replace('<code>', '<font name="Courier"><b>').replace('</code>', '</b></font>')
                                
                                # Handle headers with proper formatting
                                html_content = html_content.replace('<h1>', '<b><font size="14">').replace('</h1>', '</font></b><br/><br/>')
                                html_content = html_content.replace('<h2>', '<b><font size="12">').replace('</h2>', '</font></b><br/><br/>')
                                html_content = html_content.replace('<h3>', '<b><font size="11">').replace('</h3>', '</font></b><br/>')
                                html_content = html_content.replace('<h4>', '<b>').replace('</h4>', '</b><br/>')
                                
                                # Handle lists with better formatting
                                html_content = html_content.replace('<ul>', '<br/>').replace('</ul>', '<br/>')
                                html_content = html_content.replace('<ol>', '<br/>').replace('</ol>', '<br/>')
                                html_content = html_content.replace('<li>', '  • ').replace('</li>', '<br/>')
                                
                                # Handle blockquotes
                                html_content = html_content.replace('<blockquote>', '<i>"').replace('</blockquote>', '"</i><br/>')
                                
                                # Handle image tags - remove them or replace with text since ReportLab can't handle them easily
                                # Replace img tags with [Image] placeholder or remove them entirely
                                html_content = re.sub(r'<img[^>]*alt=["\']([^"\'][^>]*?)["\'][^>]*/?>', r'[Image: \1]', html_content)
                                html_content = re.sub(r'<img[^>]*src=["\']([^"\'][^>]*?)["\'][^>]*/?>', r'[Image]', html_content)
                                html_content = re.sub(r'<img[^>]*/?>', '[Image]', html_content)
                                
                                # Clean up extra line breaks
                                html_content = html_content.replace('<br/><br/><br/>', '<br/><br/>')
                                
                                if html_content.strip():
                                    try:
                                        story.append(Paragraph(html_content, content_style))
                                    except Exception as para_error:
                                        # If ReportLab's Paragraph fails to parse HTML, fall back to plain text
                                        print(f"ReportLab Paragraph error: {para_error}")
                                        # Strip HTML tags and use plain text
                                        plain_text = re.sub(r'<[^>]+>', '', html_content)
                                        story.append(Paragraph(plain_text.replace('\n', '<br/>'), content_style))
                        else:  # Code block
                            if part.strip():
                                # Extract language and code
                                lines = part.strip().split('\n')
                                language = ""
                                code_lines = lines
                                
                                # Check if first line is a language identifier
                                if lines and not any(char.isspace() for char in lines[0]) and len(lines[0]) < 20:
                                    language = lines[0]
                                    code_lines = lines[1:]
                                
                                code_text = '\n'.join(code_lines)
                                
                                # Add code block with language label if present
                                if language:
                                    story.append(Paragraph(f"<b>Code ({language}):</b>", content_style))
                                else:
                                    story.append(Paragraph("<b>Code:</b>", content_style))
                                
                                story.append(Preformatted(code_text, code_style))
                                story.append(Spacer(1, 6))  # Small space after code block
                
                else:
                    # No code blocks, process as regular markdown
                    html_content = markdown.markdown(content)
                    
                    # Clean up HTML for ReportLab with better formatting
                    html_content = html_content.replace('<p>', '').replace('</p>', '<br/><br/>')
                    html_content = html_content.replace('<strong>', '<b>').replace('</strong>', '</b>')
                    html_content = html_content.replace('<em>', '<i>').replace('</em>', '</i>')
                    html_content = html_content.replace('<code>', '<font name="Courier"><b>').replace('</code>', '</b></font>')
                    
                    # Handle headers with proper sizing
                    html_content = html_content.replace('<h1>', '<b><font size="14">').replace('</h1>', '</font></b><br/><br/>')
                    html_content = html_content.replace('<h2>', '<b><font size="12">').replace('</h2>', '</font></b><br/><br/>')
                    html_content = html_content.replace('<h3>', '<b><font size="11">').replace('</h3>', '</font></b><br/>')
                    html_content = html_content.replace('<h4>', '<b>').replace('</h4>', '</b><br/>')
                    
                    # Handle lists with better formatting
                    html_content = html_content.replace('<ul>', '<br/>').replace('</ul>', '<br/>')
                    html_content = html_content.replace('<ol>', '<br/>').replace('</ol>', '<br/>')
                    html_content = html_content.replace('<li>', '  • ').replace('</li>', '<br/>')
                    
                    # Handle blockquotes
                    html_content = html_content.replace('<blockquote>', '<i>"').replace('</blockquote>', '"</i><br/>')
                    
                    # Handle image tags - remove them or replace with text since ReportLab can't handle them easily
                    # Replace img tags with [Image] placeholder or remove them entirely
                    html_content = re.sub(r'<img[^>]*alt=["\']([^"\'][^>]*?)["\'][^>]*/?>', r'[Image: \1]', html_content)
                    html_content = re.sub(r'<img[^>]*src=["\']([^"\'][^>]*?)["\'][^>]*/?>', r'[Image]', html_content)
                    html_content = re.sub(r'<img[^>]*/?>', '[Image]', html_content)
                    
                    # Clean up extra line breaks
                    html_content = html_content.replace('<br/><br/><br/>', '<br/><br/>')
                    
                    if html_content.strip():
                        try:
                            story.append(Paragraph(html_content, content_style))
                        except Exception as para_error:
                            # If ReportLab's Paragraph fails to parse HTML, fall back to plain text
                            print(f"ReportLab Paragraph error: {para_error}")
                            # Strip HTML tags and use plain text
                            plain_text = re.sub(r'<[^>]+>', '', html_content)
                            story.append(Paragraph(plain_text.replace('\n', '<br/>'), content_style))
                    
            except Exception as e:
                # Fallback to plain text if markdown processing fails
                print(f"Error processing markdown: {e}")
                story.append(Paragraph(content.replace('\n', '<br/>'), content_style))
            
            # Add space between messages
            if i < len(messages) - 1:
                story.append(Spacer(1, 15))
        
        # Build PDF
        doc.build(story)
        
        # Add cleanup task to background tasks
        def cleanup_file():
            try:
                os.unlink(filepath)
            except:
                pass
        
        background_tasks.add_task(cleanup_file)
        
        # Return file for download
        return FileResponse(
            filepath,
            media_type='application/pdf',
            filename=filename,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Improved FastAPI streaming chat server...")
    print("Open http://localhost:8001 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8001)
