#!/usr/bin/env python3
"""
TTLCache-based context manager for OfflineLM
Provides persistent context across all LLM modes (Ollama, Groq, RAG, Search)
"""

from cachetools import TTLCache
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Tuple
import threading
import uuid
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class SearchContext:
    """Context from search operations"""
    query: str
    results: List[Dict[str, Any]]
    timestamp: datetime
    session_id: str
    result_count: int
    
    def to_dict(self):
        return {
            'query': self.query,
            'results': self.results,
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'result_count': self.result_count
        }

@dataclass  
class RagContext:
    """Context from RAG operations"""
    query: str
    chunks: List[str]
    source_text: str
    timestamp: datetime
    session_id: str
    chunk_count: int
    
    def to_dict(self):
        return {
            'query': self.query,
            'chunks': self.chunks,
            'source_text': self.source_text[:500] + "..." if len(self.source_text) > 500 else self.source_text,
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'chunk_count': self.chunk_count
        }

class ConversationContextCache:
    """
    Session-scoped context cache using TTL for automatic cleanup.
    Stores search results and RAG contexts that persist across mode toggles.
    """
    
    def __init__(self, ttl_minutes: int = 60, maxsize: int = 1000):
        # TTL caches for quick lookup
        self._search_cache = TTLCache(maxsize=maxsize//2, ttl=ttl_minutes * 60)
        self._rag_cache = TTLCache(maxsize=maxsize//2, ttl=ttl_minutes * 60)
        
        # Session storage for accumulated contexts
        self._session_contexts = TTLCache(maxsize=100, ttl=ttl_minutes * 60)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Configuration
        self.max_search_history = 5  # Keep last 5 search contexts per session
        self.max_rag_history = 3     # Keep last 3 RAG contexts per session
        self.max_context_length = 2000  # Max chars per context in LLM prompt
        
        logger.info(f"ConversationContextCache initialized: TTL={ttl_minutes}min, maxsize={maxsize}")
    
    def _get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get or create session context storage"""
        with self._lock:
            if session_id not in self._session_contexts:
                self._session_contexts[session_id] = {
                    'search_history': [],
                    'rag_history': [],
                    'created_at': datetime.now(),
                    'last_accessed': datetime.now()
                }
                logger.debug(f"Created new session context: {session_id}")
            else:
                # Update last accessed time
                self._session_contexts[session_id]['last_accessed'] = datetime.now()
            
            return self._session_contexts[session_id]
    
    def add_search_context(self, session_id: str, query: str, search_results: List[Dict[str, Any]]) -> None:
        """
        Add search results to session context cache.
        
        Args:
            session_id: Unique session identifier
            query: The search query that was executed
            search_results: List of search result dictionaries
        """
        with self._lock:
            try:
                search_ctx = SearchContext(
                    query=query,
                    results=search_results,
                    timestamp=datetime.now(),
                    session_id=session_id,
                    result_count=len(search_results)
                )
                
                # Store in TTL cache for quick access
                search_key = f"{session_id}_search_{datetime.now().timestamp()}"
                self._search_cache[search_key] = search_ctx
                
                # Add to session history
                session_ctx = self._get_session_context(session_id)
                session_ctx['search_history'].append(search_ctx)
                
                # Keep only recent search history
                if len(session_ctx['search_history']) > self.max_search_history:
                    session_ctx['search_history'] = session_ctx['search_history'][-self.max_search_history:]
                
                logger.info(f"Added search context to session {session_id}: query='{query[:50]}...', results={len(search_results)}")
                
            except Exception as e:
                logger.error(f"Error adding search context: {e}")
    
    def add_rag_context(self, session_id: str, query: str, chunks: List[str], source_text: str = "") -> None:
        """
        Add RAG chunks to session context cache.
        
        Args:
            session_id: Unique session identifier  
            query: The query that triggered RAG retrieval
            chunks: List of retrieved text chunks
            source_text: Source text that chunks were derived from
        """
        with self._lock:
            try:
                rag_ctx = RagContext(
                    query=query,
                    chunks=chunks,
                    source_text=source_text,
                    timestamp=datetime.now(),
                    session_id=session_id,
                    chunk_count=len(chunks)
                )
                
                # Store in TTL cache
                rag_key = f"{session_id}_rag_{datetime.now().timestamp()}"
                self._rag_cache[rag_key] = rag_ctx
                
                # Add to session history
                session_ctx = self._get_session_context(session_id)
                session_ctx['rag_history'].append(rag_ctx)
                
                # Keep only recent RAG history
                if len(session_ctx['rag_history']) > self.max_rag_history:
                    session_ctx['rag_history'] = session_ctx['rag_history'][-self.max_rag_history:]
                
                logger.info(f"Added RAG context to session {session_id}: query='{query[:50]}...', chunks={len(chunks)}")
                
            except Exception as e:
                logger.error(f"Error adding RAG context: {e}")
    
    def get_available_context_for_llm(self, session_id: str, current_query: str, 
                                     include_search: bool = True, include_rag: bool = True,
                                     max_contexts: int = 3) -> str:
        """
        Get formatted context string for LLM, including relevant previous contexts.
        
        Args:
            session_id: Session to get context for
            current_query: Current user query (for relevance scoring)
            include_search: Whether to include search contexts
            include_rag: Whether to include RAG contexts  
            max_contexts: Maximum number of context items to include
            
        Returns:
            Formatted context string for LLM
        """
        with self._lock:
            try:
                session_ctx = self._get_session_context(session_id)
                context_parts = []
                total_length = 0
                
                # Add search context if available and requested
                if include_search and session_ctx['search_history']:
                    search_context = self._format_search_context(
                        session_ctx['search_history'], 
                        current_query,
                        max_contexts
                    )
                    if search_context and total_length + len(search_context) < self.max_context_length:
                        context_parts.append(f"--- Previous Search Results ---\n{search_context}")
                        total_length += len(search_context)
                
                # Add RAG context if available and requested
                if include_rag and session_ctx['rag_history'] and total_length < self.max_context_length:
                    rag_context = self._format_rag_context(
                        session_ctx['rag_history'], 
                        current_query,
                        max_contexts
                    )
                    if rag_context and total_length + len(rag_context) < self.max_context_length:
                        context_parts.append(f"--- Previous Retrieved Context ---\n{rag_context}")
                        total_length += len(rag_context)
                
                result = "\n\n".join(context_parts) if context_parts else ""
                
                if result:
                    logger.debug(f"Generated context for session {session_id}: {len(result)} chars, {len(context_parts)} sections")
                
                return result
                
            except Exception as e:
                logger.error(f"Error generating context for LLM: {e}")
                return ""
    
    def _format_search_context(self, search_history: List[SearchContext], 
                              current_query: str, max_contexts: int) -> str:
        """Format search history for LLM context with relevance scoring"""
        try:
            # Get most recent searches, prioritizing relevance
            relevant_searches = self._score_search_relevance(search_history, current_query)
            recent_searches = relevant_searches[:max_contexts]
            
            formatted = []
            for search_ctx in recent_searches:
                # Format each search with limited results
                results_summary = []
                for i, result in enumerate(search_ctx.results[:2], 1):  # Top 2 results per search
                    url = result.get('url', 'No URL')
                    title = result.get('title', 'No Title')
                    
                    # Get content - prefer llm_input_chunk, fallback to content
                    content = result.get('llm_input_chunk', '') or result.get('content', '')
                    if len(content) > 200:
                        content = content[:200] + "..."
                    
                    if content:
                        results_summary.append(f"Source {i} ({title}): {url}\n{content}")
                
                if results_summary:
                    query_time = search_ctx.timestamp.strftime("%H:%M")
                    formatted.append(f"[{query_time}] Query: {search_ctx.query}\n" + "\n".join(results_summary))
            
            return "\n\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error formatting search context: {e}")
            return ""
    
    def _format_rag_context(self, rag_history: List[RagContext], 
                           current_query: str, max_contexts: int) -> str:
        """Format RAG history for LLM context with relevance scoring"""
        try:
            # Get most relevant RAG contexts
            relevant_rag = self._score_rag_relevance(rag_history, current_query)
            recent_rag = relevant_rag[:max_contexts]
            
            formatted = []
            for rag_ctx in recent_rag:
                # Format chunks with preview
                chunks_preview = []
                for i, chunk in enumerate(rag_ctx.chunks[:2], 1):  # Top 2 chunks per RAG
                    chunk_preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
                    chunks_preview.append(f"Chunk {i}: {chunk_preview}")
                
                if chunks_preview:
                    query_time = rag_ctx.timestamp.strftime("%H:%M")
                    formatted.append(f"[{query_time}] Query: {rag_ctx.query}\n" + "\n".join(chunks_preview))
            
            return "\n\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error formatting RAG context: {e}")
            return ""
    
    def _score_search_relevance(self, search_history: List[SearchContext], 
                               current_query: str) -> List[SearchContext]:
        """Score and sort search history by relevance to current query"""
        try:
            scored_searches = []
            current_query_lower = current_query.lower()
            
            for search_ctx in search_history:
                score = 0
                
                # Recency score (newer = higher score)
                time_diff = datetime.now() - search_ctx.timestamp
                recency_score = max(0, 1 - (time_diff.total_seconds() / 3600))  # Decay over 1 hour
                
                # Query similarity score (simple keyword matching)
                query_words = set(current_query_lower.split())
                search_words = set(search_ctx.query.lower().split())
                similarity_score = len(query_words.intersection(search_words)) / max(len(query_words), 1)
                
                # Combined score
                score = (recency_score * 0.3) + (similarity_score * 0.7)
                scored_searches.append((score, search_ctx))
            
            # Sort by score (highest first)
            scored_searches.sort(key=lambda x: x[0], reverse=True)
            return [ctx for score, ctx in scored_searches]
            
        except Exception as e:
            logger.error(f"Error scoring search relevance: {e}")
            return search_history  # Fallback to original order
    
    def _score_rag_relevance(self, rag_history: List[RagContext], 
                            current_query: str) -> List[RagContext]:
        """Score and sort RAG history by relevance to current query"""
        try:
            scored_rag = []
            current_query_lower = current_query.lower()
            
            for rag_ctx in rag_history:
                score = 0
                
                # Recency score
                time_diff = datetime.now() - rag_ctx.timestamp
                recency_score = max(0, 1 - (time_diff.total_seconds() / 3600))
                
                # Query similarity score
                query_words = set(current_query_lower.split())
                rag_words = set(rag_ctx.query.lower().split())
                similarity_score = len(query_words.intersection(rag_words)) / max(len(query_words), 1)
                
                # Content relevance score (check if current query terms appear in chunks)
                content_score = 0
                for chunk in rag_ctx.chunks:
                    chunk_lower = chunk.lower()
                    for word in query_words:
                        if word in chunk_lower:
                            content_score += 1
                content_score = min(1.0, content_score / (len(query_words) * len(rag_ctx.chunks)))
                
                # Combined score
                score = (recency_score * 0.2) + (similarity_score * 0.5) + (content_score * 0.3)
                scored_rag.append((score, rag_ctx))
            
            # Sort by score (highest first)
            scored_rag.sort(key=lambda x: x[0], reverse=True)
            return [ctx for score, ctx in scored_rag]
            
        except Exception as e:
            logger.error(f"Error scoring RAG relevance: {e}")
            return rag_history  # Fallback to original order
    
    def clear_session(self, session_id: str) -> None:
        """
        Clear all context for a specific session (called when UI clear button pressed).
        This removes conversation context but system messages will be preserved by the LLM classes.
        """
        with self._lock:
            try:
                # Remove from session storage
                if session_id in self._session_contexts:
                    del self._session_contexts[session_id]
                    logger.info(f"Cleared session context: {session_id}")
                
                # Remove from TTL caches
                search_keys_to_remove = [k for k in self._search_cache.keys() if k.startswith(f"{session_id}_")]
                for key in search_keys_to_remove:
                    del self._search_cache[key]
                
                rag_keys_to_remove = [k for k in self._rag_cache.keys() if k.startswith(f"{session_id}_")]
                for key in rag_keys_to_remove:
                    del self._rag_cache[key]
                
                logger.info(f"Cleared {len(search_keys_to_remove)} search and {len(rag_keys_to_remove)} RAG cache entries for session {session_id}")
                
            except Exception as e:
                logger.error(f"Error clearing session {session_id}: {e}")
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of cached contexts for a session"""
        with self._lock:
            try:
                if session_id not in self._session_contexts:
                    return {'session_id': session_id, 'exists': False}
                
                session_ctx = self._session_contexts[session_id]
                return {
                    'session_id': session_id,
                    'exists': True,
                    'created_at': session_ctx['created_at'].isoformat(),
                    'last_accessed': session_ctx['last_accessed'].isoformat(),
                    'search_contexts': len(session_ctx['search_history']),
                    'rag_contexts': len(session_ctx['rag_history']),
                    'has_context': len(session_ctx['search_history']) > 0 or len(session_ctx['rag_history']) > 0
                }
                
            except Exception as e:
                logger.error(f"Error getting session summary: {e}")
                return {'session_id': session_id, 'exists': False, 'error': str(e)}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics for monitoring"""
        with self._lock:
            try:
                return {
                    'search_cache': {
                        'size': len(self._search_cache),
                        'maxsize': self._search_cache.maxsize,
                        'ttl': self._search_cache.ttl,
                    },
                    'rag_cache': {
                        'size': len(self._rag_cache), 
                        'maxsize': self._rag_cache.maxsize,
                        'ttl': self._rag_cache.ttl,
                    },
                    'sessions': {
                        'active_sessions': len(self._session_contexts),
                        'maxsize': self._session_contexts.maxsize,
                        'ttl': self._session_contexts.ttl,
                    },
                    'memory_usage': {
                        'max_search_history': self.max_search_history,
                        'max_rag_history': self.max_rag_history,
                        'max_context_length': self.max_context_length,
                    }
                }
                
            except Exception as e:
                logger.error(f"Error getting cache stats: {e}")
                return {'error': str(e)}

# Global cache instance - TTL of 45 minutes, max 1000 items
context_cache = ConversationContextCache(ttl_minutes=45, maxsize=1000)

# Convenience functions for easy access
def add_search_context(session_id: str, query: str, results: List[Dict]) -> None:
    """Add search context to global cache"""
    context_cache.add_search_context(session_id, query, results)

def add_rag_context(session_id: str, query: str, chunks: List[str], source: str = "") -> None:
    """Add RAG context to global cache"""
    context_cache.add_rag_context(session_id, query, chunks, source)

def get_context_for_llm(session_id: str, query: str, include_search: bool = True, 
                       include_rag: bool = True) -> str:
    """Get available context for LLM"""
    return context_cache.get_available_context_for_llm(session_id, query, include_search, include_rag)

def clear_session_context(session_id: str) -> None:
    """Clear all context for a session"""
    context_cache.clear_session(session_id)

def get_session_info(session_id: str) -> Dict[str, Any]:
    """Get session context summary"""
    return context_cache.get_session_summary(session_id)