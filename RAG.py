#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) module for text retrieval using Ollama embeddings.
Implements recursive character text splitting and embedding-based similarity search.
"""

import os
import re
import json
import yaml
import asyncio
import aiohttp
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a text chunk with its metadata."""
    content: str
    start_index: int
    end_index: int
    embedding: Optional[np.ndarray] = None

class RecursiveCharacterTextSplitter:
    """
    Implements recursive character text splitting that tries to split on larger units
    of text first (paragraphs, lines, words) before falling back to characters.
    """
    
    def __init__(self, chunk_size: int = 5000, chunk_overlap: int = 1000):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Define separators in order of preference (largest to smallest units)
        self.separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            " ",     # Words
        ]

    def split_text(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks using recursive character splitting.
        
        Args:
            text: The input text to split
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        current_position = 0
        
        while current_position < len(text):
            # Calculate the end position for this chunk
            end_position = min(current_position + self.chunk_size, len(text))
            
            # If we're at the end of the text, just take what's left
            if end_position == len(text):
                chunk_text = text[current_position:end_position]
                chunks.append(TextChunk(
                    content=chunk_text,
                    start_index=current_position,
                    end_index=end_position
                ))
                break
            
            # Try to find a good split point using separators
            proposed_split = self._find_split_point(text, current_position, end_position)
            # Snap to a word boundary to avoid splitting words in half
            split_point = self._snap_to_word_boundary(text, current_position, end_position, proposed_split)
            
            chunk_text = text[current_position:split_point]
            chunks.append(TextChunk(
                content=chunk_text,
                start_index=current_position,
                end_index=split_point
            ))
            
            # Move to the next position with overlap
            current_position = max(split_point - self.chunk_overlap, current_position + 1)
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def _find_split_point(self, text: str, start: int, end: int) -> int:
        """
        Find the best split point within the range using separators.
        
        Args:
            text: The full text
            start: Start position
            end: End position
            
        Returns:
            The best split point
        """
        # Try each separator in order of preference
        for separator in self.separators:
            if separator == "":
                # Fall back to character splitting
                return end
            
            # Look for the separator near the end position
            # Search backwards from the end to find the last occurrence
            search_start = max(start, end - (self.chunk_size // 4))  # Don't go too far back
            
            for i in range(end - len(separator), search_start - 1, -1):
                if text[i:i + len(separator)] == separator:
                    return i + len(separator)
        
        # If no separator found, split at the character level
        return end

    def _is_word_char(self, ch: str) -> bool:
        """Return True if character is considered part of a word (alnum or underscore)."""
        return ch.isalnum() or ch == "_"

    def _snap_to_word_boundary(self, text: str, start: int, end: int, proposed: int) -> int:
        """
        Adjust the proposed split point so we don't split a word in half.

        Strategy:
        - If proposed is already at a boundary (or at text edges), keep it.
        - Else, search backward to the nearest non-word char; if found, split after it.
        - Else, search forward up to a small cap for the next non-word char and split after it.
        - If none found, keep proposed to ensure progress.
        """
        if proposed <= start:
            return min(end, len(text))
        if proposed >= len(text):
            return len(text)

        # If boundary between word/non-word, it's safe
        left_is_word = self._is_word_char(text[proposed - 1])
        right_is_word = self._is_word_char(text[proposed]) if proposed < len(text) else False
        if not (left_is_word and right_is_word):
            return proposed

        # Search backward for a non-word character between start and proposed
        for i in range(proposed - 1, start - 1, -1):
            if not self._is_word_char(text[i]):
                # split right after the non-word char
                adjusted = i + 1
                # Ensure we don't return start (would create empty chunk)
                return adjusted if adjusted > start else min(end, len(text))

        # If none backward, search forward up to a cap for a non-word character
        forward_cap = min(len(text) - 1, end + max(50, self.chunk_size // 10))
        for j in range(proposed, forward_cap + 1):
            if not self._is_word_char(text[j]):
                return j + 1

        # Give up: return proposed to ensure progress
        return proposed

class TextRetriever:
    """
    Text retrieval system using Ollama embeddings and recursive character splitting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the text retriever with configuration.
        
        Args:
            config: Configuration dictionary containing text_retriever settings
        """
        retriever_config = config.get('text_retriever', {})
        
        self.embedding_model = retriever_config.get('embedding_model', 'nomic-embed-text')
        self.chunk_size = retriever_config.get('chunk_size', 5000)
        self.chunk_overlap = retriever_config.get('chunk_overlap', 1000)
        self.top_k = retriever_config.get('top_k', 5)
        self.ollama_endpoint = "http://localhost:11434"
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        self.chunks: List[TextChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        
        logger.info(f"Initialized TextRetriever with model: {self.embedding_model}")
    
    async def process_text(self, text: str) -> None:
        """
        Async version: process a large text by splitting it into chunks and
        generating embeddings.

        Args:
            text: The input text to process
        """
        logger.info(f"Processing text of {len(text)} characters")

        # Split text into chunks
        self.chunks = self.text_splitter.split_text(text)

        # Generate embeddings for all chunks (await within the running loop)
        await self._generate_embeddings()
    
    async def _generate_embeddings(self) -> None:
        """Generate embeddings for all chunks using Ollama."""
        if not self.chunks:
            logger.warning("No chunks to generate embeddings for")
            return
        
        logger.info(f"Generating embeddings for {len(self.chunks)} chunks")
        
        embeddings_list = []
        
        async with aiohttp.ClientSession() as session:
            for i, chunk in enumerate(self.chunks):
                try:
                    embedding = await self._get_embedding(session, chunk.content)
                    chunk.embedding = embedding
                    embeddings_list.append(embedding)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Generated embeddings for {i + 1}/{len(self.chunks)} chunks")
                        
                except Exception as e:
                    logger.error(f"Failed to generate embedding for chunk {i}: {e}")
                    # Use zero embedding as fallback
                    zero_embedding = np.zeros(384)  # nomic-embed-text dimension
                    chunk.embedding = zero_embedding
                    embeddings_list.append(zero_embedding)
        
        # Convert to numpy array for efficient similarity computation
        self.embeddings = np.array(embeddings_list)
        logger.info(f"Generated embeddings matrix of shape: {self.embeddings.shape}")
    
    async def _get_embedding(self, session: aiohttp.ClientSession, text: str) -> np.ndarray:
        """
        Get embedding for a text using Ollama API.
        
        Args:
            session: aiohttp session
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        url = f"{self.ollama_endpoint}/api/embeddings"
        payload = {
            "model": self.embedding_model,
            "prompt": text
        }
        
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return np.array(data['embedding'])
            else:
                error_text = await response.text()
                raise Exception(f"Ollama API error {response.status}: {error_text}")
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return np.dot(a, b) / (norm_a * norm_b)
    
    async def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[TextChunk, float]]:
        """
        Retrieve the most similar chunks for a given query.
        
        Args:
            query: The search query
            top_k: Number of top chunks to return (uses config default if None)
            
        Returns:
            List of tuples containing (chunk, similarity_score) sorted by similarity
        """
        if not self.chunks or self.embeddings is None:
            logger.warning("No processed text available for retrieval")
            return []
        
        k = top_k if top_k is not None else self.top_k
        
        logger.info(f"Retrieving top {k} chunks for query: '{query[:50]}...'")
        
        # Get query embedding
        async with aiohttp.ClientSession() as session:
            try:
                query_embedding = await self._get_embedding(session, query)
            except Exception as e:
                logger.error(f"Failed to generate query embedding: {e}")
                return []
        
        # Calculate similarities
        similarities = []
        for i, chunk in enumerate(self.chunks):
            if chunk.embedding is not None:
                similarity = self.cosine_similarity(query_embedding, chunk.embedding)
                similarities.append((chunk, similarity))
        
        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:k]
        
        logger.info(f"Retrieved {len(top_results)} chunks with similarities: {[f'{s:.3f}' for _, s in top_results]}")
        
        return top_results
    
    def get_context_string(self, retrieved_chunks: List[Tuple[TextChunk, float]]) -> str:
        """
        Convert retrieved chunks into a formatted context string.
        
        Args:
            retrieved_chunks: List of (chunk, similarity) tuples
            
        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, (chunk, similarity) in enumerate(retrieved_chunks, 1):
            context_parts.append(f"--- Context {i} (similarity: {similarity:.3f}) ---")
            context_parts.append(chunk.content.strip())
            context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            logger.info(f"Loading configuration from {config_path}")
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {
            'text_retriever': {
                'embedding_model': 'nomic-embed-text',
                'chunk_size': 5000,
                'chunk_overlap': 1000,
                'top_k': 2
            }
        }




def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyPDF2 if available.

    Returns an empty string on failure or if PyPDF2 is not installed.
    """
    try:
        import PyPDF2
    except Exception:
        logger.warning("PyPDF2 not installed or failed to import; cannot read PDF files")
        return ""

    try:
        text_parts = []
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        return "\n\n".join(text_parts).strip()
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
        return ""


def load_sample_text(pdf_path: Optional[str] = None) -> str:
    """Load sample text from a PDF if given and available, otherwise return the embedded sample text.

    Args:
        pdf_path: Optional path to a PDF file to extract text from.
    """
    if pdf_path:
        try:
            if os.path.exists(pdf_path):
                extracted = extract_text_from_pdf(pdf_path)
                if extracted:
                    logger.info(f"Loaded sample text from PDF: {pdf_path}")
                    return extracted
                else:
                    logger.warning(f"No text extracted from PDF {pdf_path}; using embedded sample")
            else:
                logger.info(f"PDF not found at {pdf_path}; using embedded sample")
        except Exception as e:
            logger.error(f"Error while attempting to load PDF {pdf_path}: {e}")

    return EMBEDDED_SAMPLE_TEXT


# Embedded sample text used as a fallback when no PDF is provided or extraction fails.
EMBEDDED_SAMPLE_TEXT = """
Artificial Intelligence (AI) is a broad field of computer science focused on creating systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions based on that data.

Deep Learning is a specialized branch of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. It has been particularly successful in areas like image recognition, natural language processing, and speech recognition.

Natural Language Processing (NLP) is another important area of AI that focuses on the interaction between computers and human language. It enables computers to understand, interpret, and generate human language in a valuable way.

Computer Vision is the field of AI that enables computers to interpret and understand visual information from the world around them. This includes tasks like object detection, facial recognition, and scene understanding.

Reinforcement Learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment and receiving rewards or penalties based on those actions.

The applications of AI are vast and growing, including autonomous vehicles, medical diagnosis, financial trading, recommendation systems, virtual assistants, and many other domains that benefit from intelligent automation.
"""


# -------------------- Ollama model warm-up helpers --------------------
async def load_ollama_models(models: List[str], endpoint: str = "http://localhost:11434", keepalive_minutes: int = 30) -> Dict[str, Dict[str, Any]]:
    """
    Ask Ollama to load the given models and keep them alive for the specified duration.

    This function posts to Ollama's APIs to trigger a model load and sets keep-alive
    to reduce cold start latency for subsequent requests.

    Args:
        models: List of Ollama model names (e.g., ["llama3", "nomic-embed-text"]).
        endpoint: Ollama server endpoint (default http://localhost:11434).
        keepalive_minutes: Keep-alive duration in minutes (default 30).

    Returns:
        Dict mapping model name -> {"ok": bool, "method": str, "error": Optional[str]}
    """
    keep_alive = f"{keepalive_minutes}m"

    async def _warm_model(session: aiohttp.ClientSession, model: str) -> Tuple[str, Dict[str, Any]]:
        # Try /api/generate first (works for chat/instruct models)
        gen_url = f"{endpoint}/api/generate"
        gen_payload = {
            "model": model,
            "prompt": " ",        # minimal no-op prompt
            "stream": False,
            "keep_alive": keep_alive,
        }
        try:
            async with session.post(gen_url, json=gen_payload) as resp:
                if resp.status == 200:
                    # Model warmed via generate
                    return model, {"ok": True, "method": "generate", "error": None}
                # If not supported (e.g., embedding-only), try embeddings API
        except Exception as e:
            gen_error = str(e)
        else:
            gen_error = await resp.text()

        # Fallback: /api/embeddings (works for embedding models)
        emb_url = f"{endpoint}/api/embeddings"
        emb_payload = {
            "model": model,
            "prompt": "warm up",
            "keep_alive": keep_alive,
        }
        try:
            async with session.post(emb_url, json=emb_payload) as resp2:
                if resp2.status == 200:
                    return model, {"ok": True, "method": "embeddings", "error": None}
                emb_error = await resp2.text()
        except Exception as e2:
            emb_error = str(e2)

        return model, {"ok": False, "method": "none", "error": f"generate_error={gen_error}; embeddings_error={emb_error}"}

    results: Dict[str, Dict[str, Any]] = {}
    async with aiohttp.ClientSession() as session:
        tasks = [_warm_model(session, m) for m in models]
        for model, res in await asyncio.gather(*tasks):
            results[model] = res

    return results


def load_ollama_models_sync(models: List[str], endpoint: str = "http://localhost:11434", keepalive_minutes: int = 30) -> Dict[str, Dict[str, Any]]:
    """
    Synchronous wrapper for load_ollama_models for convenience in non-async contexts.
    WARNING: Do not call this from within an existing event loop.
    """
    return asyncio.run(load_ollama_models(models, endpoint=endpoint, keepalive_minutes=keepalive_minutes))

async def main():

    

    """
    Example usage of the TextRetriever class.
    """
    print("=== TextRetriever Usage Example ===")
    
    # Load configuration
    config = load_config()
    
    # Initialize retriever
    retriever = TextRetriever(config)
    
    # Load sample text from `sample.pdf` if present; otherwise use embedded sample text.
    sample_text = load_sample_text(r'/home/nitish/Downloads/Papers/ML Papers/randomforest2001.pdf')

    print(f"Processing sample text ({len(sample_text)} characters)...")

    # Process the text (async)
    await retriever.process_text(sample_text)
    
    print(f"Text split into {len(retriever.chunks)} chunks")
    
    # Example queries
    queries = ["Explain the random forest algorithm"]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        
        # Retrieve relevant chunks
        results = await retriever.retrieve(query, top_k=7)
        
        if results:
            print("Retrieved chunks:")
            for i, (chunk, similarity) in enumerate(results, 1):
                print(f"\n{i}. Similarity: {similarity:.3f}")
                print(f"   Content: {chunk.content[:]}... \n\n\n\n\n")
            
            # Get formatted context
            context = retriever.get_context_string(results)
            print(f"\nFormatted context length: {len(context)} characters")
        else:
            print("No relevant chunks found.")

if __name__ == "__main__":
    asyncio.run(main())