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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, blue, darkgray, HexColor
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from bs4 import BeautifulSoup
import markdown
from basic_ollama_agent_with_post import OllamaChat, GroqChat
from search_tool import TavilySearcher
from typing import List, Dict, Optional
import threading
import time
import subprocess
import logging
import yaml

# Set up logging for prompt length tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
prompt_logger = logging.getLogger('prompt_tracker')

def log_prompt_length(context: str, content: str, model: str = "unknown"):
    """Log the length of prompts being sent to LLMs"""
    char_count = len(content)
    estimated_tokens = char_count // 4
    prompt_logger.info(f"🔍 {context} | Model: {model} | Characters: {char_count} | Est. Tokens: {estimated_tokens}")
    if estimated_tokens > 5000:
        prompt_logger.warning(f"⚠️  High token count detected: {estimated_tokens} tokens - may exceed model limits")
    return estimated_tokens

# Configuration loading
def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Config file not found at {config_path}, using defaults")
        return {
            'thinking_words': [
                ["<think>", "</think>"],
                ["<analysis>", "</analysis>"],
                ["Thinking...", "...done thinking"]
            ]
        }
    except Exception as e:
        print(f"Error loading config: {e}, using defaults")
        return {
            'thinking_words': [
                ["<think>", "</think>"],
                ["<analysis>", "</analysis>"],
                ["Thinking...", "...done thinking"]
            ]
        }

# Load configuration
config = load_config()
thinking_patterns = config.get('thinking_words', [
    ["<think>", "</think>"],
    ["<analysis>", "</analysis>"],
    ["Thinking...", "...done thinking"]
])
import re
import asyncio

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
groq_chat_agent = None

# Define available Groq models
GROQ_MODELS = [
    {
        'name': 'openai/gpt-oss-120b',
        'displayName': 'GPT-OSS 120B',
        'provider': 'groq',
        'size': '120B',
        'modified_at': 'Available'
    },
    {
        'name': 'openai/gpt-oss-20b',
        'displayName': 'GPT-OSS 20B',
        'provider': 'groq',
        'size': '20B',
        'modified_at': 'Available'
    },
    {
        'name': 'qwen/qwen3-32b',
        'displayName': 'Qwen3 32B',
        'provider': 'groq',
        'size': '32B',
        'modified_at': 'Available'
    },
    {
        'name': 'meta-llama/llama-4-scout-17b-16e-instruct',
        'displayName': 'Llama-4 Scout 17B',
        'provider': 'groq',
        'size': '17B',
        'modified_at': 'Available'
    },
    {
        'name': 'meta-llama/llama-4-maverick-17b-128e-instruct',
        'displayName': 'Llama-4 Maverick 17B',
        'provider': 'groq',
        'size': '17B',
        'modified_at': 'Available'
    },
    {
        'name': 'llama-3.3-70b-versatile',
        'displayName': 'Llama 3.3 70B Versatile',
        'provider': 'groq',
        'size': '70B',
        'modified_at': 'Available'
    },
    {
        'name': 'deepseek-r1-distill-llama-70b',
        'displayName': 'DeepSeek R1 Distill Llama 70B',
        'provider': 'groq',
        'size': '70B',
        'modified_at': 'Available'
    }
]

def is_groq_model(model_name: str) -> bool:
    """Check if a model name corresponds to a Groq model."""
    groq_model_names = [model['name'] for model in GROQ_MODELS]
    return model_name in groq_model_names

def get_chat_agent(model_name: str, search_enabled: bool = False):
    """Get the appropriate chat agent (OllamaChat or GroqChat) for the given model."""
    global chat_agent, groq_chat_agent
    
    # Get the appropriate system message based on mode
    system_mode = 'search_mode' if search_enabled else 'chat_mode'
    system_message = config.get('system_instructions', {}).get(system_mode, None)
    
    if is_groq_model(model_name):
        if groq_chat_agent is None or groq_chat_agent.model != model_name:
            # Create new GroqChat agent or switch model
            groq_chat_agent = GroqChat(model=model_name, system_message=system_message)
        return groq_chat_agent
    else:
        if chat_agent is None or chat_agent.model != model_name:
            # Create new OllamaChat agent or switch model
            chat_agent = OllamaChat(model=model_name, enable_thinking=True, system_message=system_message)
        return chat_agent

# Track active streaming sessions
active_streams = {}
stream_locks = {}

# Message formatting functions (migrated from frontend)
def process_think_tags(content: str) -> str:
    """
    Process thinking tags by converting them to styled HTML format.
    Supports multiple thinking patterns from configuration (config.yaml).
    Also handles new Ollama thinking format (preserves existing formatting).
    Migrated from frontend JavaScript to backend Python.
    """
    
    # If content already contains Ollama thinking format, preserve it
    if '<div class="think-section' in content and '💭 Reasoning:' in content:
        return content
    
    # Check all configured thinking patterns
    for start_marker, end_marker in thinking_patterns:
        # Escape regex special characters for pattern matching
        escaped_start = re.escape(start_marker)
        escaped_end = re.escape(end_marker)
        
        # Create regex pattern for complete thinking sections
        think_regex = rf'^({escaped_start})([\s\S]*?)({escaped_end})([\s\S]*)$'
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
        
        # Check for in-progress thinking (start marker present but no end marker)
        if content.startswith(start_marker) and end_marker not in content:
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

def format_message(content: str, skip_thinking_processing: bool = False) -> str:
    """
    Format message content with markdown processing, LaTeX math, and syntax highlighting.
    Migrated from frontend JavaScript to backend Python.
    
    Args:
        content: The content to format
        skip_thinking_processing: If True, skip thinking tag processing (for Ollama native thinking)
    """
    try:
        # Process think tags first before markdown parsing (unless skipped for Ollama native thinking)
        if skip_thinking_processing:
            processed_content = content  # Skip regex-based thinking processing
        else:
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
        
        # LaTeX math expressions are handled by frontend KaTeX rendering
        # No backend math processing needed
        
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
        if skip_thinking_processing:
            return content.replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;').replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
        else:
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
    """Initialize the chat agents with the first available model."""
    global default_model, chat_agent, groq_chat_agent
    default_model = get_first_available_model()
    chat_agent = OllamaChat(model=default_model, enable_thinking=True)
    print(f"Initialized Ollama chat agent with model: {default_model}")
    
    # Initialize Groq agent with default model (can be changed later)
    try:
        groq_chat_agent = GroqChat(model="llama-3.3-70b-versatile")
        print("Initialized Groq chat agent with default model: llama-3.3-70b-versatile")
    except Exception as e:
        print(f"Could not initialize Groq chat agent: {e}")
        groq_chat_agent = None
    
    return default_model

# Initialize the chat agents with the first available model
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

@app.get("/config/thinking-patterns")
async def get_thinking_patterns():
    """Get configured thinking patterns for frontend consistency."""
    return {
        "thinking_patterns": thinking_patterns,
        "message": "Thinking patterns loaded from configuration"
    }

@app.get("/models")
async def get_available_models():
    """Get list of available models categorized by provider (Ollama and Groq)."""
    ollama_models = []
    
    try:
        # Try to get models from Ollama API
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            for model in data.get('models', []):
                ollama_models.append({
                    'name': model.get('name', 'Unknown'),
                    'provider': 'ollama',
                    'size': model.get('size', 'Unknown'),
                    'modified_at': model.get('modified_at', 'Unknown')
                })
    except Exception as e:
        print(f"Error fetching models from Ollama: {e}")
        # Fallback to common Ollama models if API fails
        ollama_models = [
            {"name": default_model, "provider": "ollama", "size": "Unknown", "modified_at": "Recently"},
            {"name": "qwen2.5:7b", "provider": "ollama", "size": "4.7GB", "modified_at": "Recently"},
            {"name": "gemma3:4b-it-fp16", "provider": "ollama", "size": "2.4GB", "modified_at": "Recently"},
            {"name": "llama3.2:3b", "provider": "ollama", "size": "2.0GB", "modified_at": "Recently"},
            {"name": "phi3:mini", "provider": "ollama", "size": "2.3GB", "modified_at": "Recently"},
        ]
    
    # Get Groq models (always available if API key is configured)
    groq_models = GROQ_MODELS.copy()
    
    # Combine all models for backward compatibility
    all_models = ollama_models + groq_models
    
    return {
        "models": all_models,
        "ollama_models": ollama_models,
        "groq_models": groq_models,
        "default_model": default_model,
        "status": "success"
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
    """Change the active model, supporting both Ollama and Groq models."""
    try:
        global chat_agent, groq_chat_agent
        
        print(f"\n\n Model Requested: {request.model}")
        print(f"Model Type: {'Groq' if is_groq_model(request.model) else 'Ollama'}")
        
        stopped_models = []
        
        if is_groq_model(request.model):
            # Handle Groq model change
            print(f"Switching to Groq model: {request.model}")
            
            # Create new GroqChat agent with the selected model
            groq_chat_agent = GroqChat(model=request.model)
            
            # Restore conversation history if provided
            if request.conversation_history:
                groq_chat_agent.conversation_history = request.conversation_history
            
            print(f"Groq model changed to {groq_chat_agent.model} with history restored: {len(request.conversation_history)} messages")
            
            return {
                "status": "success",
                "message": f"Model changed to {request.model} (Groq)",
                "history_restored": len(request.conversation_history) > 0,
                "stopped_models": [],
                "provider": "groq"
            }
        else:
            # Handle Ollama model change
            print(f"Switching to Ollama model: {request.model}")
            
            # Step 1: Stop all other Ollama models
            stopped_models = await stop_all_other_models(request.model)
            
            # Step 2: Ensure the new Ollama model is loaded with keepalive
            model_loaded = await ensure_model_loaded_with_keepalive(request.model)
            
            if not model_loaded:
                return {
                    "status": "error", 
                    "message": f"Failed to load Ollama model {request.model}",
                    "stopped_models": stopped_models
                }
            
            # Step 3: Create new OllamaChat agent with the selected model
            chat_agent = OllamaChat(model=request.model, enable_thinking=True)
            
            # Step 4: Restore conversation history if provided
            if request.conversation_history:
                chat_agent.conversation_history = request.conversation_history
            
            print(f"Ollama model changed to {chat_agent.model} with history restored: {len(request.conversation_history)} messages")
            print(f"Stopped models: {stopped_models}")
            
            return {
                "status": "success", 
                "message": f"Model changed to {request.model} (Ollama)",
                "history_restored": len(request.conversation_history) > 0,
                "stopped_models": stopped_models,
                "keepalive_set": "30m",
                "provider": "ollama"
            }
            
    except Exception as e:
        print(f"Error changing model: {e}")
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
async def chat_stream_sse(message: str, model: str = None, session: str = None, search_enabled: bool = False, search_count: int = 5):
    """Stream chat responses using Server-Sent Events with EventSource."""
    
    # Optional debug logging (uncomment for debugging)
    # print(f"DEBUG: Stream request - search_enabled={search_enabled}, search_count={search_count}, message='{message[:50]}...'")
    
    async def generate():
        try:
            # Register this streaming session
            if session:
                active_streams[session] = True
                stream_locks[session] = threading.Lock()
            
            # Get the appropriate chat agent based on model type
            if model:
                current_agent = get_chat_agent(model, search_enabled)
                # Preserve conversation history from the main agents
                if is_groq_model(model):
                    if groq_chat_agent and groq_chat_agent.model == model:
                        current_agent = groq_chat_agent
                    else:
                        # Create new GroqChat agent and preserve history
                        system_mode = 'search_mode' if search_enabled else 'chat_mode'
                        system_message = config.get('system_instructions', {}).get(system_mode, None)
                        current_agent = GroqChat(model=model, system_message=system_message)
                        # Copy history from either groq or ollama agent
                        if groq_chat_agent:
                            current_agent.conversation_history = groq_chat_agent.conversation_history.copy()
                        elif chat_agent:
                            current_agent.conversation_history = chat_agent.conversation_history.copy()
                else:
                    if chat_agent and chat_agent.model == model:
                        current_agent = chat_agent
                    else:
                        # Create new OllamaChat agent and preserve history
                        current_agent = OllamaChat(model=model, enable_thinking=True)
                        # Copy history from either ollama or groq agent
                        if chat_agent:
                            current_agent.conversation_history = chat_agent.conversation_history.copy()
                        elif groq_chat_agent:
                            current_agent.conversation_history = groq_chat_agent.conversation_history.copy()
            else:
                # Use default agent (Ollama)
                current_agent = chat_agent if chat_agent else OllamaChat(model=default_model, enable_thinking=True)
            
            # Check if search is enabled and perform search if needed
            final_message = message
            raw_search_data = None
            if search_enabled:
                # print(f"DEBUG: Search enabled - attempting search for: {message[:50]}...")  # Debug line
                try:
                    # Check if API key is available
                    import os
                    api_key = os.getenv("TAVILY_API_KEY")
                    if not api_key:
                        yield f"data: {json.dumps({'type': 'search_error', 'message': 'Search disabled: TAVILY_API_KEY not configured. Proceeding without search...'})}\n\n"
                        # print("DEBUG: Search skipped - no API key")  # Debug line
                    else:
                        yield f"data: {json.dumps({'type': 'search_start', 'message': f'Searching ({search_count} sites) and thinking...'})}\n\n"
                        
                        # Initialize TavilySearcher with config values
                        search_config = config.get('search', {})
                        searcher = TavilySearcher(
                            max_results=search_count,
                            include_raw_content=search_config.get('include_raw_content', True),
                            chunks_per_source=search_config.get('chunks_per_source', 2)
                        )
                        
                        # Get raw search results for the frontend
                        raw_search_results = await searcher.search(
                            query=message,
                            max_results=search_count
                        )
                        
                        # Store raw search data for frontend display with exact LLM input chunks
                        if "error" not in raw_search_results:
                            raw_search_data = {
                                'query': message,



                                'results': []
                            }
                            
                            # Format the exact chunks that will be sent to LLM (mirroring search_wrapper_for_llm_input logic)
                            for i, result in enumerate(raw_search_results.get('results', []), 1):
                                # Use the same logic as search_wrapper_for_llm_input to determine content
                                raw_content = result.get('raw_content', '')
                                content = result.get('content', '')
                                url = result.get('url', 'No URL')
                                title = result.get('title', '')
                                
                                # Use raw content if config allows it AND it's not too long, otherwise use snippet
                                if search_config.get('include_raw_content', True) and raw_content and len(raw_content) < 2000:
                                    used_content = raw_content
                                elif content:
                                    used_content = content
                                else:
                                    continue
                                
                                if used_content:
                                    # Create the exact context entry that will be sent to LLM
                                    if title:
                                        formatted_content = f"**{title}**\n{used_content}"
                                    else:
                                        formatted_content = used_content
                                    
                                    # Add source information exactly as in search_wrapper_for_llm_input
                                    chunk_header = f"\n--- Source {i}: {url} ---\n"
                                    formatted_chunk = f"{chunk_header}{formatted_content}"
                                    
                                    raw_search_data['results'].append({
                                        'url': url,
                                        'llm_input_chunk': formatted_chunk  # The exact text sent to LLM for this result
                                    })
                        
                        # Perform search using search_wrapper_for_llm_input for LLM
                        search_results = await searcher.search_wrapper_for_llm_input(
                            query=message,
                            max_results=search_count
                        )
                        
                        # Close the searcher
                        await searcher.close()
                        
                        # Use search results as the message to LLM
                        final_message = search_results
                        
                        # Log the search results length before sending to LLM
                        log_prompt_length("SEARCH_CONTEXT", final_message, model or "default")
                        
                        yield f"data: {json.dumps({'type': 'search_complete', 'message': 'Search complete, generating response...'})}\n\n"
                        # print(f"DEBUG: Search completed successfully - context length: {len(search_results)}")  # Debug line
                    
                except Exception as search_error:
                    print(f"Search error: {search_error}")
                    yield f"data: {json.dumps({'type': 'search_error', 'message': f'Search failed: {str(search_error)}. Proceeding without search...'})}\n\n"
                    # Continue with original message if search fails
                    final_message = message
            # else:
                # print(f"DEBUG: Search disabled by user")  # Debug line
            
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting response...'})}\n\n"
            
            # Stream with interruption checking and backend formatting
            full_response = ""
            try:
                # Check if we need special handling for search to preserve conversation history
                if search_enabled and final_message != message:
                    # For search: Store original user message in history, but send search context to LLM
                    # First manually add the original user message to conversation history
                    current_agent.conversation_history.append({'role': 'user', 'content': message})
                    
                    # Now we need to call the LLM with the search context but not store it in history
                    # We'll use a direct API call approach instead of chat_stream to avoid history pollution
                    
                    if hasattr(current_agent, 'model') and is_groq_model(current_agent.model):
                        # Handle Groq model - use existing client with API key
                        client = current_agent.client  # Use the existing Groq client that has the API key
                        
                        # Prepare messages for Groq API call
                        messages = current_agent.conversation_history.copy()
                        messages[-1]['content'] = final_message  # Use search context for API call
                        
                        # Log total message content length for Groq
                        total_content = "\n".join([msg.get('content', '') for msg in messages])
                        log_prompt_length("GROQ_API_CALL", total_content, current_agent.model)
                        
                        try:
                            stream = client.chat.completions.create(
                                model=current_agent.model,
                                messages=messages,
                                temperature=0.7,
                                stream=True
                            )
                            
                            for chunk in stream:
                                if chunk.choices[0].delta.content is not None:
                                    content_chunk = chunk.choices[0].delta.content
                                    full_response += content_chunk
                                    
                                    # Check if stream should be stopped
                                    if session and not active_streams.get(session, False):
                                        print(f"Stream {session} was stopped by user")
                                        yield f"data: {json.dumps({'type': 'stopped', 'message': 'Stream stopped by user'})}\n\n"
                                        break
                                    
                                    # Format and yield the chunk (skip thinking for Groq as it uses tag-based)
                                    try:
                                        formatted_content = format_message(full_response, skip_thinking_processing=False)  # Groq uses tag-based thinking
                                        yield f"data: {json.dumps({'type': 'chunk', 'content': content_chunk, 'formatted_content': formatted_content, 'raw_content': full_response})}\n\n"
                                    except Exception as format_error:
                                        print(f"Formatting error: {format_error}")
                                        yield f"data: {json.dumps({'type': 'chunk', 'content': content_chunk, 'raw_content': full_response})}\n\n"
                                    
                                    time.sleep(0.01)
                        except Exception as groq_error:
                            print(f"Groq API error: {groq_error}")
                            yield f"data: {json.dumps({'type': 'error', 'message': f'Groq API error: {str(groq_error)}'})}\n\n"
                    
                    else:
                        # Handle Ollama model
                        import requests
                        
                        # Prepare messages for Ollama API call
                        messages = current_agent.conversation_history.copy()
                        messages[-1]['content'] = final_message  # Use search context for API call
                        
                        # Log total message content length for Ollama
                        total_content = "\n".join([msg.get('content', '') for msg in messages])
                        log_prompt_length("OLLAMA_API_CALL", total_content, current_agent.model)
                        
                        # Prepare request parameters
                        request_params = {
                            "model": current_agent.model,
                            "messages": messages,
                            "stream": True,
                            "keep_alive": "30m"
                        }
                        
                        # Try with thinking first, fallback to without thinking if it fails
                        thinking_enabled = False
                        if hasattr(current_agent, 'enable_thinking') and current_agent.enable_thinking:
                            try:
                                # Test request with thinking
                                test_params = request_params.copy()
                                test_params.update({"think": True, "stream": False, "messages": [{"role": "user", "content": "test"}]})
                                test_response = requests.post("http://localhost:11434/api/chat", json=test_params, timeout=5)
                                if test_response.status_code == 200:
                                    # Model accepts think parameter - enable thinking
                                    request_params["think"] = True
                                    thinking_enabled = True
                                    print(f"✅ Using thinking for model {current_agent.model} - will use message.thinking")
                                else:
                                    print(f"⚠️  Model {current_agent.model} returned status {test_response.status_code}, using standard mode")
                            except Exception as e:
                                print(f"⚠️  Thinking not supported for model {current_agent.model}, using standard mode: {e}")
                        
                        print(f"🔧 Final configuration for {current_agent.model}: thinking_enabled={thinking_enabled}")
                        
                        try:
                            response = requests.post(
                                "http://localhost:11434/api/chat",
                                json=request_params,
                                stream=True
                            )
                            
                            thinking_content = ""
                            thinking_displayed = False
                            
                            for line in response.iter_lines():
                                if line:
                                    try:
                                        chunk_data = json.loads(line.decode('utf-8'))
                                        message_data = chunk_data.get('message', {})
                                        
                                        # Handle thinking content (only if thinking is enabled for this model)
                                        if thinking_enabled and 'thinking' in message_data:
                                            thinking_chunk = message_data['thinking']
                                            if thinking_chunk:
                                                thinking_content += thinking_chunk
                                                print(f"🧠 Got thinking chunk for {current_agent.model}: {len(thinking_chunk)} chars")
                                                
                                                # Display thinking section start
                                                if not thinking_displayed and thinking_content.strip():
                                                    thinking_displayed = True
                                                    print(f"🧠 Starting thinking display for {current_agent.model}")
                                                    yield f"data: {json.dumps({'type': 'chunk', 'content': '<div class=\"think-section think-streaming\"><em>💭 Reasoning:\\n'})}\n\n"
                                                
                                                # Yield thinking chunk
                                                yield f"data: {json.dumps({'type': 'chunk', 'content': thinking_chunk})}\n\n"
                                        
                                        # Handle main content
                                        if 'content' in message_data:
                                            content_chunk = message_data['content']
                                            if content_chunk:
                                                # Close thinking section if we're starting main content
                                                if thinking_displayed and not full_response:
                                                    yield f"data: {json.dumps({'type': 'chunk', 'content': '</em></div>\\n\\n'})}\n\n"
                                                
                                                full_response += content_chunk
                                                
                                                # Check if stream should be stopped
                                                if session and not active_streams.get(session, False):
                                                    print(f"Stream {session} was stopped by user")
                                                    yield f"data: {json.dumps({'type': 'stopped', 'message': 'Stream stopped by user'})}\n\n"
                                                    break
                                                
                                                # Format and yield the chunk (skip thinking processing for Ollama native thinking)
                                                try:
                                                    formatted_content = format_message(full_response, skip_thinking_processing=thinking_enabled)
                                                    yield f"data: {json.dumps({'type': 'chunk', 'content': content_chunk, 'formatted_content': formatted_content, 'raw_content': full_response})}\n\n"
                                                except Exception as format_error:
                                                    print(f"Formatting error: {format_error}")
                                                    yield f"data: {json.dumps({'type': 'chunk', 'content': content_chunk, 'raw_content': full_response})}\n\n"
                                                
                                                time.sleep(0.01)
                                        
                                        if chunk_data.get('done', False):
                                            break
                                    except json.JSONDecodeError:
                                        continue
                        except Exception as ollama_error:
                            print(f"Ollama API error: {ollama_error}")
                            yield f"data: {json.dumps({'type': 'error', 'message': f'Ollama API error: {str(ollama_error)}'})}\n\n"
                    
                    # Manually add assistant response to conversation history
                    if full_response:
                        current_agent.conversation_history.append({'role': 'assistant', 'content': full_response})
                
                else:
                    # Normal flow for non-search requests
                    # Log the regular message length
                    log_prompt_length("REGULAR_CHAT", final_message, current_agent.model if current_agent else "unknown")
                    
                    for chunk in current_agent.chat_stream(final_message):
                        # Check if stream should be stopped
                        if session and not active_streams.get(session, False):
                            print(f"Stream {session} was stopped by user")
                            # Send stop signal to OLLAMA only (Groq handles stopping automatically)
                            if not is_groq_model(current_agent.model):
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
                        
                        # Format the accumulated response using backend formatting (skip thinking for Ollama)
                        try:
                            # Check if this is an Ollama model to skip thinking processing
                            is_ollama = not (hasattr(current_agent, 'model') and is_groq_model(current_agent.model))
                            formatted_content = format_message(full_response, skip_thinking_processing=is_ollama)
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
                                # Check if this is an Ollama model to skip thinking processing
                                is_ollama = not (hasattr(current_agent, 'model') and is_groq_model(current_agent.model))
                                final_formatted = format_message(full_response, skip_thinking_processing=is_ollama)
                                stats = calculate_chat_stats()
                                response_data = {'type': 'done', 'message': 'Response complete', 'final_formatted': final_formatted, 'stats': stats}
                                if raw_search_data:
                                    response_data['raw_search_data'] = raw_search_data
                                yield f"data: {json.dumps(response_data)}\n\n"
                            except Exception as final_format_error:
                                print(f"Final formatting error: {final_format_error}")
                                response_data = {'type': 'done', 'message': 'Response complete'}
                                if raw_search_data:
                                    response_data['raw_search_data'] = raw_search_data
                                yield f"data: {json.dumps(response_data)}\n\n"
                
                # Send final completion message for both search and non-search flows
                if session and active_streams.get(session, False):
                    try:
                        # Check if this is an Ollama model to skip thinking processing
                        is_ollama = not (hasattr(current_agent, 'model') and is_groq_model(current_agent.model))
                        final_formatted = format_message(full_response, skip_thinking_processing=is_ollama)
                        stats = calculate_chat_stats()
                        response_data = {'type': 'done', 'message': 'Response complete', 'final_formatted': final_formatted, 'stats': stats}
                        if raw_search_data:
                            response_data['raw_search_data'] = raw_search_data
                        yield f"data: {json.dumps(response_data)}\n\n"
                    except Exception as final_format_error:
                        print(f"Final formatting error: {final_format_error}")
                        response_data = {'type': 'done', 'message': 'Response complete'}
                        if raw_search_data:
                            response_data['raw_search_data'] = raw_search_data
                        yield f"data: {json.dumps(response_data)}\n\n"
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
    
    # Create async generator and return StreamingResponse
    async def async_generate():
        async for chunk in generate():
            yield chunk
    
    return StreamingResponse(
        async_generate(),
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
            # Log prompt length for POST endpoint
            log_prompt_length("POST_CHAT", chat_message.message, chat_agent.model if chat_agent else "unknown")
            
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
            # Check if we're using an Ollama model to skip thinking processing
            is_ollama = not (hasattr(chat_agent, 'model') and is_groq_model(chat_agent.model))
            formatted_content = format_message(raw_content, skip_thinking_processing=is_ollama)
            
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
            
            # Log prompt length for minimal chat endpoint
            log_prompt_length("MINIMAL_CHAT", chat_message.message, chat_agent.model if chat_agent else "unknown")
            
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

# Professional color scheme
DARK_STEEL_BLUE = HexColor('#4682B4')
PROFESSIONAL_BLACK = HexColor('#000000')

def register_fonts():
    """Register fonts for PDF generation with fallback support."""
    try:
        # Try to register Segoe UI (Windows/modern systems)
        # On Linux, we'll try common font paths
        font_paths = [
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',  # Linux common
            '/System/Library/Fonts/Helvetica.ttc',  # macOS
            'C:/Windows/Fonts/segoeui.ttf',  # Windows
        ]
        
        registered_font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    if font_path.endswith('segoeui.ttf'):
                        pdfmetrics.registerFont(TTFont('SegoeUI', font_path))
                        registered_font = 'SegoeUI'
                    elif font_path.endswith('LiberationSans-Regular.ttf'):
                        pdfmetrics.registerFont(TTFont('SegoeUI', font_path))  # Use Liberation as Segoe UI substitute
                        registered_font = 'SegoeUI'
                    break
                except:
                    continue
        
        # Try Calibri as fallback
        if not registered_font:
            calibri_paths = [
                'C:/Windows/Fonts/calibri.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
            ]
            
            for font_path in calibri_paths:
                if os.path.exists(font_path):
                    try:
                        pdfmetrics.registerFont(TTFont('Calibri', font_path))
                        registered_font = 'Calibri'
                        break
                    except:
                        continue
        
        return registered_font or 'Helvetica'  # Final fallback to built-in
        
    except Exception as e:
        print(f"Font registration failed: {e}")
        return 'Helvetica'

def parse_html_table(table_html):
    """Parse HTML table and convert to ReportLab Table data."""
    soup = BeautifulSoup(table_html, 'html.parser')
    tables = soup.find_all('table')
    
    if not tables:
        return None
    
    # Process the first table found
    table = tables[0]
    rows = table.find_all('tr')
    
    if not rows:
        return None
    
    data = []
    for row in rows:
        cells = row.find_all(['td', 'th'])
        row_data = []
        for cell in cells:
            # Get text content and preserve some formatting
            cell_text = cell.get_text(strip=True)
            if cell.name == 'th':
                cell_text = f"<b>{cell_text}</b>"
            row_data.append(cell_text)
        data.append(row_data)
    
    return data

def create_professional_table(table_data, primary_font='Helvetica', available_width=None):
    """Create a professionally styled ReportLab Table that fits within page boundaries with text wrapping."""
    if not table_data:
        return None
    
    # Calculate available width (now 7.5 inches for letter size with 0.5 inch margins)
    if available_width is None:
        available_width = 7.5 * inch
    
    # Wrap text in table cells to prevent overflow
    def wrap_cell_text(text, max_width_chars=50):
        """Wrap text in table cells to fit within column width."""
        if not text or len(str(text)) <= max_width_chars:
            return str(text)
        
        # Break long words and wrap text
        import textwrap
        wrapped_lines = textwrap.fill(str(text), width=max_width_chars, break_long_words=True)
        return wrapped_lines
    
    # Process table data to wrap text in cells
    wrapped_table_data = []
    for row in table_data:
        wrapped_row = []
        for cell in row:
            # Determine max characters based on column count and available width
            num_cols = len(row)
            max_chars_per_col = max(20, min(80, int(120 / num_cols)))  # Adaptive character limit
            wrapped_cell = wrap_cell_text(cell, max_chars_per_col)
            wrapped_row.append(wrapped_cell)
        wrapped_table_data.append(wrapped_row)
    
    # Calculate column widths based on content and available space
    if wrapped_table_data:
        num_cols = len(wrapped_table_data[0])
        # Use more of the available width with smaller margins
        col_width = available_width / num_cols
        col_widths = [col_width] * num_cols
        
        # Ensure minimum column width for readability
        min_col_width = 1.0 * inch
        if any(w < min_col_width for w in col_widths):
            col_widths = [max(w, min_col_width) for w in col_widths]
            # Recalculate if needed
            total_width = sum(col_widths)
            if total_width > available_width:
                scale_factor = available_width / total_width
                col_widths = [w * scale_factor for w in col_widths]
    
    # Create the table with calculated widths and wrapped data
    table = Table(wrapped_table_data, colWidths=col_widths)
    
    # Apply professional styling with enhanced text wrapping
    table_style = TableStyle([
        # Header row styling (first row)
        ('BACKGROUND', (0, 0), (-1, 0), DARK_STEEL_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), PROFESSIONAL_BLACK),
        ('FONTNAME', (0, 0), (-1, 0), primary_font),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTWEIGHT', (0, 0), (-1, 0), 'BOLD'),
        
        # Data rows styling
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#FFFFFF')),
        ('TEXTCOLOR', (0, 1), (-1, -1), PROFESSIONAL_BLACK),
        ('FONTNAME', (0, 1), (-1, -1), primary_font),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        
        # Border and alignment
        ('GRID', (0, 0), (-1, -1), 1, PROFESSIONAL_BLACK),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        
        # Padding for better text wrapping
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        
        # Alternating row colors for better readability
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#FFFFFF'), HexColor('#F8F9FA')]),
        
        # Enhanced text wrapping and overflow handling
        ('WORDWRAP', (0, 0), (-1, -1), True),
        ('SPLITLONGWORDS', (0, 0), (-1, -1), True),
    ])
    
    table.setStyle(table_style)
    return table

def process_content_with_tables(content, primary_font='Helvetica'):
    """Process content and extract tables, returning both regular content and table objects."""
    # Convert markdown to HTML first
    html_content = markdown.markdown(content, extensions=['tables'])
    
    # Parse with BeautifulSoup to handle tables
    soup = BeautifulSoup(html_content, 'html.parser')
    
    story_elements = []
    
    # Process each element in order
    for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'blockquote', 'table', 'pre', 'code']):
        if element.name == 'table':
            # Extract table data and create ReportLab table
            rows = element.find_all('tr')
            if rows:
                table_data = []
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    row_data = []
                    for cell in cells:
                        cell_text = cell.get_text(strip=True)
                        row_data.append(cell_text)
                    table_data.append(row_data)
                
                if table_data:
                    table_obj = create_professional_table(table_data, primary_font, available_width=7.5*inch)
                    if table_obj:
                        story_elements.append(('table', table_obj))
        else:
            # Handle other elements as HTML text
            element_html = str(element)
            story_elements.append(('html', element_html))
    
    # If no specific elements found, process as regular content
    if not story_elements:
        story_elements.append(('html', html_content))
    
    return story_elements

def process_html_for_reportlab(html_content):
    """Process HTML content for ReportLab compatibility with professional formatting."""
    # Handle math expressions first to prevent character splitting
    # Look for common LaTeX patterns and preserve them as single units
    import re
    
    # Preserve inline math expressions (between $ $ or \( \))
    math_patterns = [
        (r'\$([^$]+)\$', r'<i>\1</i>'),  # $expression$ -> italic
        (r'\\?\(([^)]+)\\\)', r'<i>\1</i>'),  # \(expression\) -> italic
        (r'\\?\[([^]]+)\\\]', r'<i>\1</i>'),  # \[expression\] -> italic
    ]
    
    for pattern, replacement in math_patterns:
        html_content = re.sub(pattern, replacement, html_content)
    
    # Clean up HTML tags for ReportLab
    html_content = html_content.replace('<p>', '').replace('</p>', '<br/><br/>')
    html_content = html_content.replace('<strong>', '<b>').replace('</strong>', '</b>')
    html_content = html_content.replace('<em>', '<i>').replace('</em>', '</i>')
    html_content = html_content.replace('<code>', '<font name="Courier"><b>').replace('</code>', '</b></font>')
    
    # Handle headers with 2-tier font system (header size vs body size)
    html_content = html_content.replace('<h1>', '<b><font size="14">').replace('</h1>', '</font></b><br/><br/>')
    html_content = html_content.replace('<h2>', '<b><font size="13">').replace('</h2>', '</font></b><br/><br/>')
    html_content = html_content.replace('<h3>', '<b><font size="12">').replace('</h3>', '</font></b><br/>')
    html_content = html_content.replace('<h4>', '<b><font size="11">').replace('</h4>', '</font></b><br/>')
    
    # Handle lists with professional formatting
    html_content = html_content.replace('<ul>', '<br/>').replace('</ul>', '<br/>')
    html_content = html_content.replace('<ol>', '<br/>').replace('</ol>', '<br/>')
    html_content = html_content.replace('<li>', '  • ').replace('</li>', '<br/>')
    
    # Handle blockquotes professionally
    html_content = html_content.replace('<blockquote>', '<i>"').replace('</blockquote>', '"</i><br/>')
    
    # Remove/replace images with professional placeholders
    html_content = re.sub(r'<img[^>]*alt=["\']([^"\'][^>]*?)["\'][^>]*/?>', r'[Image: \1]', html_content)
    html_content = re.sub(r'<img[^>]*src=["\']([^"\'][^>]*?)["\'][^>]*/?>', r'[Image]', html_content)
    html_content = re.sub(r'<img[^>]*/?>', '[Image]', html_content)
    
    # Clean up extra line breaks but preserve intentional spacing
    html_content = html_content.replace('<br/><br/><br/>', '<br/><br/>')
    
    # Fix math expressions that might have been split
    # Remove unwanted line breaks in mathematical expressions
    html_content = re.sub(r'<i>([^<]+)</i>', lambda m: '<i>' + m.group(1).replace('\n', ' ').replace('<br/>', ' ') + '</i>', html_content)
    
    return html_content

def process_latex_for_pdf(content):
    """
    Process LaTeX expressions for PDF generation while preserving formatting.
    Converts LaTeX to more readable format without breaking into separate lines.
    """
    # Handle LaTeX expressions more carefully to avoid line breaks
    import re
    
    # Replace common LaTeX patterns with readable text, keeping them as single units
    latex_replacements = [
        (r'\$\$([^$]+)\$\$', r'[\1]'),  # Display math -> [expression]
        (r'\$([^$]+)\$', r'\1'),        # Inline math -> expression
        (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)'),  # Fractions
        (r'\\sqrt\{([^}]+)\}', r'sqrt(\1)'),              # Square roots
        (r'\\sum_?\{?([^}]*)\}?', r'sum(\1)'),            # Summations
        (r'\\int_?\{?([^}]*)\}?', r'integral(\1)'),       # Integrals
        (r'\\alpha', 'α'),
        (r'\\beta', 'β'),
        (r'\\gamma', 'γ'),
        (r'\\delta', 'δ'),
        (r'\\pi', 'π'),
        (r'\\theta', 'θ'),
        (r'\\lambda', 'λ'),
        (r'\\mu', 'μ'),
        (r'\\sigma', 'σ'),
    ]
    
    for pattern, replacement in latex_replacements:
        content = re.sub(pattern, replacement, content)
    
    return content

def process_think_tags_for_pdf(content):
    """
    Process thinking tags for PDF generation by converting them to readable format.
    Supports multiple thinking patterns from configuration (config.yaml).
    Removes HTML formatting and converts to markdown-friendly format.
    """
    import re
    
    # First process LaTeX expressions
    content = process_latex_for_pdf(content)
    
    # Check all configured thinking patterns
    for start_marker, end_marker in thinking_patterns:
        # Escape regex special characters for pattern matching
        escaped_start = re.escape(start_marker)
        escaped_end = re.escape(end_marker)
        
        # Create regex pattern for complete thinking sections
        think_regex = rf'^({escaped_start})([\s\S]*?)({escaped_end})([\s\S]*)$'
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
    """Generate and download chat conversation as PDF with professional formatting and table support."""
    try:
        # Register fonts with Segoe UI preference and Calibri fallback
        primary_font = register_fonts()
        print(f"Using font: {primary_font}")
        
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
        
        # Create PDF document with reduced margins (0.5 inch on sides)
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=1.2*inch,
            bottomMargin=1*inch
        )
        
        # Professional style definitions with 2-tier font system
        styles = getSampleStyleSheet()
        
        # Header font size (for titles, headers)
        HEADER_FONT_SIZE = 14
        # Body font size (for content)
        BODY_FONT_SIZE = 10
        
        title_style = ParagraphStyle(
            'ProfessionalTitle',
            parent=styles['Heading1'],
            fontSize=22,  # Larger for main title
            fontName=primary_font,
            textColor=PROFESSIONAL_BLACK,
            spaceAfter=24,
            alignment=1,  # Center alignment
            spaceBefore=0
        )
        
        metadata_style = ParagraphStyle(
            'ProfessionalMetadata',
            parent=styles['Normal'],
            fontSize=BODY_FONT_SIZE,
            fontName=primary_font,
            textColor=DARK_STEEL_BLUE,
            spaceAfter=8,
            leftIndent=0
        )
        
        user_header_style = ParagraphStyle(
            'UserHeader',
            parent=styles['Heading2'],
            fontSize=HEADER_FONT_SIZE,  # Header tier font size
            fontName=primary_font,
            textColor=DARK_STEEL_BLUE,
            spaceBefore=18,
            spaceAfter=6,
            leftIndent=0,
            fontWeight='BOLD'  # Make explicitly bold
        )
        
        assistant_header_style = ParagraphStyle(
            'AssistantHeader',
            parent=styles['Heading2'],
            fontSize=HEADER_FONT_SIZE,  # Header tier font size
            fontName=primary_font,
            textColor=PROFESSIONAL_BLACK,
            spaceBefore=18,
            spaceAfter=6,
            leftIndent=0,
            fontWeight='BOLD'  # Make explicitly bold
        )
        
        content_style = ParagraphStyle(
            'ProfessionalContent',
            parent=styles['Normal'],
            fontSize=BODY_FONT_SIZE,  # Body tier font size
            fontName=primary_font,
            textColor=PROFESSIONAL_BLACK,
            leftIndent=12,
            spaceAfter=10,
            spaceBefore=2,
            leading=14,
            allowWidows=0,
            allowOrphans=0
        )
        
        code_style = ParagraphStyle(
            'ProfessionalCode',
            parent=styles['Code'],
            fontSize=BODY_FONT_SIZE - 1,  # Slightly smaller than body
            fontName='Courier',
            textColor=PROFESSIONAL_BLACK,
            leftIndent=20,
            spaceAfter=12,
            spaceBefore=6,
            backgroundColor=HexColor('#F8F9FA'),
            borderWidth=1,
            borderColor=DARK_STEEL_BLUE,
            borderPadding=8
        )
        
        # Build PDF content
        story = []
        
        # Professional title
        story.append(Paragraph("Chat Conversation Export", title_style))
        story.append(Spacer(1, 16))
        
        # Metadata with professional formatting
        now = datetime.now()
        story.append(Paragraph(f"<b>Export Date:</b> {now.strftime('%B %d, %Y at %I:%M %p')}", metadata_style))
        story.append(Paragraph(f"<b>AI Model:</b> {chat_agent.model}", metadata_style))
        story.append(Paragraph(f"<b>Message Count:</b> {len(messages)}", metadata_style))
        story.append(Spacer(1, 24))
        
        # Process each message with enhanced table support
        for i, message in enumerate(messages):
            # Bold, bigger message headers with proper 2-tier font system
            if message['type'] == 'user':
                story.append(Paragraph(f'<b><font size="{HEADER_FONT_SIZE}">User</font></b>', user_header_style))
            else:
                story.append(Paragraph(f'<b><font size="{HEADER_FONT_SIZE}">Assistant</font></b>', assistant_header_style))
            
            # Process content with full table and formatting support
            content = message['content']
            
            try:
                # Handle think tags specially for PDF generation
                content = process_think_tags_for_pdf(content)
                
                # Process content with tables and mixed formatting
                if '```' in content:
                    # Handle code blocks separately from table processing
                    parts = content.split('```')
                    
                    for j, part in enumerate(parts):
                        if j % 2 == 0:  # Regular content (may include tables)
                            if part.strip():
                                # Process with table support
                                story_elements = process_content_with_tables(part.strip(), primary_font)
                                
                                for element_type, element in story_elements:
                                    if element_type == 'table':
                                        story.append(Spacer(1, 8))
                                        story.append(element)
                                        story.append(Spacer(1, 8))
                                    elif element_type == 'html':
                                        # Process HTML content
                                        processed_html = process_html_for_reportlab(element)
                                        if processed_html.strip():
                                            try:
                                                story.append(Paragraph(processed_html, content_style))
                                            except Exception as para_error:
                                                print(f"ReportLab Paragraph error: {para_error}")
                                                plain_text = re.sub(r'<[^>]+>', '', processed_html)
                                                story.append(Paragraph(plain_text.replace('\n', '<br/>'), content_style))
                        
                        else:  # Code block
                            if part.strip():
                                # Extract language and code
                                lines = part.strip().split('\n')
                                language = ""
                                code_lines = lines
                                
                                if lines and not any(char.isspace() for char in lines[0]) and len(lines[0]) < 20:
                                    language = lines[0]
                                    code_lines = lines[1:]
                                
                                code_text = '\n'.join(code_lines)
                                
                                # Add code block with professional styling
                                if language:
                                    story.append(Paragraph(f"<b>Code ({language})</b>", content_style))
                                else:
                                    story.append(Paragraph("<b>Code</b>", content_style))
                                
                                story.append(Preformatted(code_text, code_style))
                
                else:
                    # Process content without code blocks but with table support
                    story_elements = process_content_with_tables(content, primary_font)
                    
                    for element_type, element in story_elements:
                        if element_type == 'table':
                            story.append(Spacer(1, 8))
                            story.append(element)
                            story.append(Spacer(1, 8))
                        elif element_type == 'html':
                            # Process HTML content with professional formatting
                            processed_html = process_html_for_reportlab(element)
                            if processed_html.strip():
                                try:
                                    story.append(Paragraph(processed_html, content_style))
                                except Exception as para_error:
                                    print(f"ReportLab Paragraph error: {para_error}")
                                    plain_text = re.sub(r'<[^>]+>', '', processed_html)
                                    story.append(Paragraph(plain_text.replace('\n', '<br/>'), content_style))
                    
            except Exception as e:
                # Fallback to plain text if processing fails
                print(f"Error processing content: {e}")
                story.append(Paragraph(content.replace('\n', '<br/>'), content_style))
            
            # Add professional spacing between messages
            if i < len(messages) - 1:
                story.append(Spacer(1, 20))
        
        # Build PDF with professional formatting
        doc.build(story)
        
        # Add cleanup task
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
    uvicorn.run(app, host="0.0.0.0", port=8002)
