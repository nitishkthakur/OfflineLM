import httpx
import asyncio
from typing import Optional, Dict, List
from io import BytesIO
import re

# Method 1: Using PyPDF2 (simple, good for basic text extraction)
async def get_arxiv_text_pypdf2(arxiv_url: str) -> Dict[str, any]:
    """
    Extract text from arXiv PDF using PyPDF2.
    Install: pip install PyPDF2
    """
    import PyPDF2
    
    try:
        # Get PDF bytes
        arxiv_id = extract_arxiv_id(arxiv_url)
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(pdf_url)
            response.raise_for_status()
            pdf_bytes = response.content
        
        # Extract text
        pdf_file = BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_content = ""
        for page_num, page in enumerate(pdf_reader.pages):
            text_content += f"\n--- Page {page_num + 1} ---\n"
            text_content += page.extract_text()
        
        return {
            "status": "success",
            "text": text_content,
            "num_pages": len(pdf_reader.pages),
            "word_count": len(text_content.split()),
            "method": "PyPDF2"
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Method 2: Using pdfplumber (better for complex layouts, tables)
async def get_arxiv_text_pdfplumber(arxiv_url: str) -> Dict[str, any]:
    """
    Extract text from arXiv PDF using pdfplumber.
    Install: pip install pdfplumber
    Better for complex layouts and tables.
    """
    import pdfplumber
    
    try:
        # Get PDF bytes
        arxiv_id = extract_arxiv_id(arxiv_url)
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(pdf_url)
            response.raise_for_status()
            pdf_bytes = response.content
        
        # Extract text
        pdf_file = BytesIO(pdf_bytes)
        text_content = ""
        tables = []
        
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text_content += f"\n--- Page {page_num + 1} ---\n"
                text_content += page.extract_text() or ""
                
                # Extract tables if any
                page_tables = page.extract_tables()
                if page_tables:
                    for table in page_tables:
                        tables.append({
                            "page": page_num + 1,
                            "table": table
                        })
        
        return {
            "status": "success",
            "text": text_content,
            "num_pages": len(pdf.pages),
            "word_count": len(text_content.split()),
            "tables": tables,
            "method": "pdfplumber"
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Method 3: Using pymupdf (fitz) - fastest and most accurate
async def get_arxiv_text_pymupdf(arxiv_url: str) -> Dict[str, any]:
    """
    Extract text from arXiv PDF using PyMuPDF (fitz).
    Install: pip install pymupdf
    Fastest and most accurate for academic papers.
    """
    import fitz  # PyMuPDF
    
    try:
        # Get PDF bytes
        arxiv_id = extract_arxiv_id(arxiv_url)
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(pdf_url)
            response.raise_for_status()
            pdf_bytes = response.content
        
        # Extract text
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_content = ""
        metadata = pdf_document.metadata
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text_content += f"\n--- Page {page_num + 1} ---\n"
            text_content += page.get_text()
        
        pdf_document.close()
        
        return {
            "status": "success",
            "text": text_content,
            "num_pages": pdf_document.page_count,
            "word_count": len(text_content.split()),
            "metadata": metadata,
            "method": "PyMuPDF"
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Method 4: Using pdfminer (most detailed, good for research papers)
async def get_arxiv_text_pdfminer(arxiv_url: str) -> Dict[str, any]:
    """
    Extract text from arXiv PDF using pdfminer.
    Install: pip install pdfminer.six
    Best for detailed text extraction with layout analysis.
    """
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams
    
    try:
        # Get PDF bytes
        arxiv_id = extract_arxiv_id(arxiv_url)
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(pdf_url)
            response.raise_for_status()
            pdf_bytes = response.content
        
        # Extract text with layout parameters
        pdf_file = BytesIO(pdf_bytes)
        laparams = LAParams(
            char_margin=2.0,
            line_margin=0.5,
            word_margin=0.1
        )
        
        text_content = extract_text(pdf_file, laparams=laparams)
        
        return {
            "status": "success",
            "text": text_content,
            "word_count": len(text_content.split()),
            "method": "pdfminer"
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Utility functions
def extract_arxiv_id(url: str) -> Optional[str]:
    """Extract arXiv ID from URL"""
    patterns = [
        r'arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)',
        r'arxiv\.org/pdf/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)\.pdf',
        r'arxiv:([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)',
        r'^([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url.strip())
        if match:
            return match.group(1)
    return None

def clean_text_for_llm(text: str) -> str:
    """
    Clean extracted text for better LLM processing.
    """
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Fix common PDF extraction issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between joined words
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words across lines
    
    # Remove page headers/footers (basic)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not (line.isdigit() or len(line) < 3):  # Skip page numbers and very short lines
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

# Main function for your chat interface
async def process_arxiv_paper_for_llm(arxiv_url: str, method: str = "pymupdf") -> Dict[str, any]:
    """
    Process arXiv paper and prepare text for LLM.
    
    Args:
        arxiv_url: arXiv URL
        method: extraction method ("pymupdf", "pdfplumber", "pypdf2", "pdfminer")
    
    Returns:
        Dict with extracted text and metadata
    """
    extraction_methods = {
        "pymupdf": get_arxiv_text_pymupdf,
        "pdfplumber": get_arxiv_text_pdfplumber,
        "pypdf2": get_arxiv_text_pypdf2,
        "pdfminer": get_arxiv_text_pdfminer
    }
    
    if method not in extraction_methods:
        return {"status": "error", "error": f"Unknown method: {method}"}
    
    # Extract text
    result = await extraction_methods[method](arxiv_url)
    
    if result["status"] == "success":
        # Clean text for LLM
        cleaned_text = clean_text_for_llm(result["text"])
        result["cleaned_text"] = cleaned_text
        result["cleaned_word_count"] = len(cleaned_text.split())
        
        # Add summary info
        result["arxiv_id"] = extract_arxiv_id(arxiv_url)
        result["ready_for_llm"] = True
    
    return result

# Example integration with your chat system
async def handle_arxiv_in_chat(arxiv_url: str, rag_mode: bool = False) -> str:
    """
    Handle arXiv paper in your chat interface.
    Returns text ready to send to OLLAMA/LLM.
    """
    try:
        # Process the paper
        result = await process_arxiv_paper_for_llm(arxiv_url, method="pymupdf")
        
        if result["status"] != "success":
            return f"Error processing arXiv paper: {result['error']}"
        
        paper_text = result["cleaned_text"]
        arxiv_id = result["arxiv_id"]
        
        if rag_mode:
            # In RAG mode, you'd add this to your vector store
            # and return a confirmation message
            # add_to_vector_store(paper_text, metadata={"source": f"arXiv:{arxiv_id}"})
            return f"ArXiv paper {arxiv_id} processed and added to knowledge base. You can now ask questions about it."
        else:
            # In regular mode, include the paper content in the prompt
            prompt = f"""
I have extracted the following research paper from arXiv (ID: {arxiv_id}):

{paper_text}

Please analyze this paper and provide insights or answer any questions about it.
"""
            return prompt
            
    except Exception as e:
        return f"Error processing arXiv paper: {str(e)}"

# Usage examples:
"""
# Basic text extraction
result = await process_arxiv_paper_for_llm("https://arxiv.org/abs/2301.00001")
if result["status"] == "success":
    llm_ready_text = result["cleaned_text"]
    print(f"Extracted {result['cleaned_word_count']} words")

# Integration with chat
llm_prompt = await handle_arxiv_in_chat(
    "https://arxiv.org/abs/2301.00001", 
    rag_mode=False
)
# Send llm_prompt to your OLLAMA model
"""