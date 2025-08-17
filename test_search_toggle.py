#!/usr/bin/env python3
"""
Test script to verify search toggle functionality
"""

import asyncio
import aiohttp
import json

async def test_search_toggle():
    """Test the search toggle functionality by making direct API calls"""
    
    base_url = "http://localhost:8001"
    
    print("Testing Search Toggle Functionality")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        
        # Test 1: Search disabled
        print("\n1. Testing with search DISABLED")
        print("-" * 30)
        
        url = f"{base_url}/chat/stream-sse"
        params = {
            "message": "What is machine learning?",
            "search_enabled": "false",
            "search_count": "5"
        }
        
        try:
            async with session.get(url, params=params) as response:
                print(f"Status: {response.status}")
                if response.status == 200:
                    # Read first few chunks
                    chunk_count = 0
                    async for line in response.content:
                        if line.startswith(b'data: '):
                            try:
                                data = json.loads(line[6:].decode())
                                print(f"Chunk {chunk_count}: {data}")
                                chunk_count += 1
                                if chunk_count >= 3:  # Only read first few chunks
                                    break
                            except:
                                pass
                else:
                    print(f"Error: {response.status}")
        except Exception as e:
            print(f"Error testing search disabled: {e}")
        
        # Test 2: Search enabled (will likely fail without API key, but we can see the attempt)
        print("\n2. Testing with search ENABLED")
        print("-" * 30)
        
        params = {
            "message": "What is machine learning?",
            "search_enabled": "true",
            "search_count": "3"
        }
        
        try:
            async with session.get(url, params=params) as response:
                print(f"Status: {response.status}")
                if response.status == 200:
                    # Read first few chunks
                    chunk_count = 0
                    async for line in response.content:
                        if line.startswith(b'data: '):
                            try:
                                data = json.loads(line[6:].decode())
                                print(f"Chunk {chunk_count}: {data}")
                                chunk_count += 1
                                if chunk_count >= 5:  # Read more chunks to see search messages
                                    break
                            except:
                                pass
                else:
                    print(f"Error: {response.status}")
        except Exception as e:
            print(f"Error testing search enabled: {e}")

if __name__ == "__main__":
    print("🔧 This test requires the FastAPI server to be running on localhost:8001")
    print("   Start the server with: python fastapi_streaming_improved.py")
    print("   Then run this test in another terminal\n")
    
    try:
        asyncio.run(test_search_toggle())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("   Make sure the server is running and accessible")