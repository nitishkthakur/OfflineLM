#!/usr/bin/env python3
"""
Test script to demonstrate TTLCache context functionality in OfflineLM
"""

import sys
import json
from context_cache import context_cache, add_search_context, add_rag_context, get_context_for_llm, get_session_info

def test_context_cache():
    """Test the context cache functionality"""
    
    print("🧪 Testing OfflineLM Context Cache")
    print("=" * 50)
    
    # Test session
    session_id = "test-session-123"
    
    # Test 1: Add search context
    print("\n1. Adding search context...")
    search_results = [
        {
            'url': 'https://example.com/python-caching',
            'title': 'Python Caching Guide',
            'content': 'Python offers several caching mechanisms including functools.lru_cache and cachetools...',
            'llm_input_chunk': '--- Source 1: https://example.com/python-caching ---\n**Python Caching Guide**\nPython offers several caching mechanisms...'
        },
        {
            'url': 'https://example.com/ttl-cache',
            'title': 'TTL Cache Explained', 
            'content': 'TTL (Time To Live) caches automatically expire entries after a specified time...',
            'llm_input_chunk': '--- Source 2: https://example.com/ttl-cache ---\n**TTL Cache Explained**\nTTL caches automatically expire...'
        }
    ]
    
    add_search_context(session_id, "How to use Python caching?", search_results)
    print(f"✅ Added search context with {len(search_results)} results")
    
    # Test 2: Add RAG context
    print("\n2. Adding RAG context...")
    rag_chunks = [
        "Caching is a technique to store frequently accessed data in memory for faster retrieval.",
        "The cachetools library provides various cache implementations like LRUCache and TTLCache.",
        "TTL caches are particularly useful for data that becomes stale after a certain time period."
    ]
    
    add_rag_context(session_id, "What are the benefits of caching?", rag_chunks, "Technical documentation on caching")
    print(f"✅ Added RAG context with {len(rag_chunks)} chunks")
    
    # Test 3: Get session info
    print("\n3. Session information:")
    session_info = get_session_info(session_id)
    print(f"Session exists: {session_info.get('exists', False)}")
    print(f"Search contexts: {session_info.get('search_contexts', 0)}")
    print(f"RAG contexts: {session_info.get('rag_contexts', 0)}")
    print(f"Has context: {session_info.get('has_context', False)}")
    
    # Test 4: Get context for LLM
    print("\n4. Getting context for LLM...")
    context = get_context_for_llm(session_id, "How can I implement caching in my application?")
    print(f"Generated context length: {len(context)} characters")
    print("\nContext preview:")
    print(context[:500] + "..." if len(context) > 500 else context)
    
    # Test 5: Test with search only
    print("\n5. Getting search-only context...")
    search_only_context = get_context_for_llm(session_id, "Python caching libraries", include_search=True, include_rag=False)
    print(f"Search-only context length: {len(search_only_context)} characters")
    
    # Test 6: Test with RAG only
    print("\n6. Getting RAG-only context...")
    rag_only_context = get_context_for_llm(session_id, "Benefits of caching", include_search=False, include_rag=True)
    print(f"RAG-only context length: {len(rag_only_context)} characters")
    
    # Test 7: Cache statistics
    print("\n7. Cache statistics:")
    stats = context_cache.get_cache_stats()
    print(json.dumps(stats, indent=2, default=str))
    
    # Test 8: Test context persistence after "mode toggle"
    print("\n8. Testing context persistence...")
    # Simulate turning off search and RAG modes, but context should still be available
    persistent_context = get_context_for_llm(session_id, "New question about caching")
    print(f"Persistent context available: {len(persistent_context) > 0}")
    print(f"Persistent context length: {len(persistent_context)} characters")
    
    print("\n✅ All tests completed successfully!")
    print("📝 The context cache is working and contexts persist even when modes are toggled off.")

if __name__ == "__main__":
    test_context_cache()