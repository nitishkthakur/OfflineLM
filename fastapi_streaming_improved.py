#!/usr/bin/env python3
"""
Improved FastAPI streaming chat with proper Server-Sent Events implementation.
"""

from fastapi import FastAPI, Request, HTTPException
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
from typing import List, Dict

app = FastAPI(title="Ollama Streaming Chat API - Improved")

# Mount static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="."), name="static")

# Initialize the chat agent
chat_agent = OllamaChat(model="qwen3:4b")

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
        with open("minimal_ui_streaming.html", "r") as f:
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
            chat_agent.conversation_history = request.conversation_history
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

@app.post("/download-chat-pdf")
async def download_chat_pdf():
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
            leading=14
        )
        
        code_style = ParagraphStyle(
            'CodeStyle',
            parent=styles['Code'],
            fontSize=9,
            leftIndent=30,
            spaceAfter=8,
            backgroundColor='#f5f5f5'
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
            
            # Process content - preserve markdown formatting
            content = message['content']
            
            # Convert markdown to HTML while preserving code blocks
            try:
                # Handle code blocks specially
                if '```' in content:
                    parts = content.split('```')
                    processed_content = ""
                    
                    for j, part in enumerate(parts):
                        if j % 2 == 0:  # Regular text
                            if part.strip():
                                # Convert markdown to HTML for regular text
                                html_content = markdown.markdown(part.strip())
                                # Clean up HTML tags for ReportLab
                                html_content = html_content.replace('<p>', '').replace('</p>', '<br/>')
                                html_content = html_content.replace('<strong>', '<b>').replace('</strong>', '</b>')
                                html_content = html_content.replace('<em>', '<i>').replace('</em>', '</i>')
                                processed_content += html_content
                        else:  # Code block
                            if part.strip():
                                # Add code block
                                lines = part.strip().split('\n')
                                # Skip language identifier if present
                                if lines and not any(char.isspace() for char in lines[0]) and len(lines[0]) < 20:
                                    lines = lines[1:]
                                code_text = '\n'.join(lines)
                                story.append(Paragraph("Code:", content_style))
                                story.append(Preformatted(code_text, code_style))
                                continue
                    
                    if processed_content.strip():
                        story.append(Paragraph(processed_content, content_style))
                
                else:
                    # No code blocks, process as regular markdown
                    html_content = markdown.markdown(content)
                    # Clean up HTML for ReportLab
                    html_content = html_content.replace('<p>', '').replace('</p>', '<br/>')
                    html_content = html_content.replace('<strong>', '<b>').replace('</strong>', '</b>')
                    html_content = html_content.replace('<em>', '<i>').replace('</em>', '</i>')
                    html_content = html_content.replace('<h1>', '<b>').replace('</h1>', '</b><br/>')
                    html_content = html_content.replace('<h2>', '<b>').replace('</h2>', '</b><br/>')
                    html_content = html_content.replace('<h3>', '<b>').replace('</h3>', '</b><br/>')
                    
                    # Handle lists
                    html_content = html_content.replace('<ul>', '').replace('</ul>', '')
                    html_content = html_content.replace('<ol>', '').replace('</ol>', '')
                    html_content = html_content.replace('<li>', '• ').replace('</li>', '<br/>')
                    
                    story.append(Paragraph(html_content, content_style))
                    
            except Exception as e:
                # Fallback to plain text if markdown processing fails
                story.append(Paragraph(content.replace('\n', '<br/>'), content_style))
            
            # Add space between messages
            if i < len(messages) - 1:
                story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        
        # Return file for download
        def cleanup_file():
            try:
                os.unlink(filepath)
            except:
                pass
        
        return FileResponse(
            filepath,
            media_type='application/pdf',
            filename=filename,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            background=cleanup_file
        )
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Improved FastAPI streaming chat server...")
    print("Open http://localhost:8001 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8001)
