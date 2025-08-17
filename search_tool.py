import asyncio
import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
try:
    from tavily import AsyncTavilyClient


except ImportError:
    print("tavily-python not installed. Install with: pip install tavily-python")
    AsyncTavilyClient = None
import os
from tavily import TavilyClient
from os import getenv
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
def tavily_to_context(search_results):
    context = []
    for result in search_results['results']:
        context.append(result['raw_content'])

    return "\n\n\n".join(context)
class TavilySearcher:
    """
    A wrapper class for Tavily search functionality with configurable defaults.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        max_results: int = 10,  # changed to valid range (0-20, default 5)
        include_raw_content: bool = True,  # should be boolean, not string
        search_depth: str = "advanced",  # default as in settings
        chunks_per_source: int = 5,  # API default value
        include_answer: bool = False,  # added default
        include_images: bool = False  # added default
    ):
        """
        Initialize TavilySearcher with default parameters.
        
        Args:
            api_key: Tavily API key. If None, will try to get from TAVILY_API_KEY env var
            max_results: Maximum number of search results (default: 10, max: 20)
            include_raw_content: Whether to include full page content (default: True)
            search_depth: Search depth level (default: "advanced")
            chunks_per_source: Max chunks per source (default: 3, only used with advanced search)
            include_answer: Whether to include generated answer (default: False)
            include_images: Whether to include images (default: False)
        """
        if AsyncTavilyClient is None:
            raise ImportError("tavily-python package is required. Install with: pip install tavily-python")
            
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Provide it directly or set TAVILY_API_KEY environment variable")
            
        self.client = AsyncTavilyClient(api_key=self.api_key)
        self.max_results = max_results
        self.include_raw_content = include_raw_content
        self.search_depth = search_depth
        self.chunks_per_source = chunks_per_source
        self.include_answer = include_answer
        self.include_images = include_images
    
    async def search(
        self, 
        query: str,
        max_results: Optional[int] = None,
        include_raw_content: Optional[bool] = None,
        search_depth: Optional[str] = None,
        chunks_per_source: Optional[int] = None,
        include_answer: Optional[bool] = None,
        include_images: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform an asynchronous search using Tavily.
        
        Args:
            query: Search query string
            max_results: Override default max_results
            include_raw_content: Override default include_raw_content (boolean)
            search_depth: Override default search_depth
            chunks_per_source: Override default chunks_per_source
            include_answer: Override default include_answer
            include_images: Override default include_images
            **kwargs: Additional parameters to pass to Tavily search
            
        Returns:
            Dictionary containing search results
        """
        search_params = {
            "query": query,
            "max_results": max_results if max_results is not None else self.max_results,
            "search_depth": search_depth if search_depth is not None else self.search_depth,
            "include_raw_content": include_raw_content if include_raw_content is not None else self.include_raw_content,
            "include_answer": include_answer if include_answer is not None else self.include_answer,
            "include_images": include_images if include_images is not None else self.include_images,
            **kwargs
        }
        
        # Only add chunks_per_source if search_depth is advanced
        if (search_depth or self.search_depth) == "advanced":
            search_params["chunks_per_source"] = chunks_per_source if chunks_per_source is not None else self.chunks_per_source
        
        # Log search parameters
        logger.info(f"Performing Tavily search with parameters: {search_params}")
        
        try:
            results = await self.client.search(**search_params)
            
            # Log search results
            num_results = len(results.get('results', []))
            logger.info(f"Search completed successfully. Found {num_results} results.")
            
            # Log individual result lengths
            for i, result in enumerate(results.get('results', []), 1):
                content = result.get('content', '')
                raw_content = result.get('raw_content', '')
                url = result.get('url', 'No URL')
                
                content_length = len(content) if content else 0
                raw_content_length = len(raw_content) if raw_content else 0
                
                logger.info(f"Result {i}: URL={url}, content_length={content_length}, raw_content_length={raw_content_length}")
            
            return results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {"error": f"Search failed: {str(e)}"}
    
    async def extract(self, urls: List[str], **kwargs) -> Dict[str, Any]:
        """Extract content from URLs."""
        try:
            return await self.client.extract(urls, **kwargs)
        except Exception as e:
            return {"error": f"Extract failed: {str(e)}"}
    
    async def search_and_extract(
        self,
        query: str,
        max_results: Optional[int] = None,
        search_depth: Optional[str] = None,
        include_answer: Optional[bool] = None,
        include_images: Optional[bool] = None,
        save_to_json: Optional[str] = None,
        **kwargs
    ) -> List[tuple[str, str]]:
        """
        Perform a search and then extract full content from each URL.
        
        Args:
            query: Search query string
            max_results: Override default max_results
            search_depth: Override default search_depth
            include_answer: Override default include_answer
            include_images: Override default include_images
            save_to_json: Optional filename to save results as JSON (default: None)
            **kwargs: Additional parameters to pass to Tavily search
            
        Returns:
            List of tuples containing (content, url) for each successfully extracted page
        """
        logger.info(f"Starting search_and_extract for query: '{query}' with max_results={max_results or self.max_results}")
        
        # First perform the search without raw content to get URLs
        search_results = await self.search(
            query=query,
            max_results=max_results,
            include_raw_content=False,  # Don't need raw content from search
            search_depth=search_depth,
            chunks_per_source=None,  # Not needed for URL extraction
            include_answer=include_answer,
            include_images=include_images,
            **kwargs
        )
        
        if "error" in search_results:
            logger.error(f"Search failed in search_and_extract: {search_results['error']}")
            return []
        
        # Extract URLs from search results
        urls = []
        url_to_title = {}  # Keep track of titles for better context
        
        for result in search_results.get('results', []):
            url = result.get('url')
            if url:
                urls.append(url)
                url_to_title[url] = result.get('title', '')
        
        logger.info(f"Extracted {len(urls)} URLs for content extraction")
        
        if not urls:
            logger.warning("No URLs found for extraction")
            return []
        
        # Extract content from all URLs
        logger.info(f"Starting content extraction from {len(urls)} URLs")
        extract_results = await self.extract(urls)
        
        if "error" in extract_results:
            logger.error(f"Extraction failed: {extract_results['error']}")
            return []
        
        # Process extraction results
        output = []
        json_data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "search_parameters": {
                "max_results": max_results or self.max_results,
                "search_depth": search_depth or self.search_depth,
                "include_answer": include_answer or self.include_answer,
                "include_images": include_images or self.include_images
            },
            "results": []
        }
        
        extracted_data = extract_results.get('results', [])
        logger.info(f"Successfully extracted content from {len(extracted_data)} pages")
        
        for i, result in enumerate(extracted_data, 1):
            url = result.get('url', '')
            content = result.get('raw_content', '') or result.get('content', '')
            
            if content and url:
                # Optionally prepend title to content for better context
                title = url_to_title.get(url, '')
                if title:
                    formatted_content = f"# {title}\n\n{content}"
                else:
                    formatted_content = content
                
                output.append((formatted_content, url))
                
                # Log extraction details
                logger.info(f"Extracted content {i}: URL={url}, content_length={len(content)}, formatted_length={len(formatted_content)}")
                
                # Add to JSON data
                json_data["results"].append({
                    "url": url,
                    "title": title,
                    "content": content,
                    "formatted_content": formatted_content,
                    "content_length": len(content)
                })
        
        # Save to JSON if requested
        if save_to_json:
            # Generate filename if not provided with extension
            if not save_to_json.endswith('.json'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{save_to_json}_{timestamp}.json"
            else:
                filename = save_to_json
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                print(f"Results saved to: {filename}")
            except Exception as e:
                print(f"Error saving JSON file: {e}")
        
        logger.info(f"search_and_extract completed. Returning {len(output)} content pieces")
        return output
    
    async def search_wrapper_for_llm_input(
        self,
        query: str,
        max_results: Optional[int] = None,
        search_depth: Optional[str] = None,
        include_answer: Optional[bool] = None,
        include_images: Optional[bool] = None,
        **kwargs
    ) -> str:
        """
        Perform a search and format results for LLM input.
        
        This method searches for content and formats it in a way that's optimized for 
        LLM consumption, providing clear context and the original question.
        Optimized for Groq models with token limits - uses search snippets only.
        
        Args:
            query: Search query string (the original user question)
            max_results: Override default max_results
            search_depth: Override default search_depth
            include_answer: Override default include_answer
            include_images: Override default include_images
            **kwargs: Additional parameters to pass to Tavily search
            
        Returns:
            Formatted string with context and question for LLM processing
        """
        logger.info(f"Starting search_wrapper_for_llm_input for query: '{query}'")
        
        # Get search results WITH raw content but limited chunks to control token usage
        search_results = await self.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_raw_content=True,  # Enable raw content for better context
            chunks_per_source=2,  # Limit chunks per source to control token usage
            include_answer=include_answer,
            include_images=include_images,
            **kwargs
        )
        
        if "error" in search_results:
            logger.error(f"Search failed in search_wrapper_for_llm_input: {search_results['error']}")
            return f"Search failed: {search_results['error']}"
        
        # Format the context from search result snippets only
        context_chunks = []
        total_content_length = 0
        
        for i, result in enumerate(search_results.get('results', []), 1):
            # Use raw_content if available, otherwise fall back to content snippet
            raw_content = result.get('raw_content', '')
            content = result.get('content', '')
            url = result.get('url', 'No URL')
            title = result.get('title', '')
            
            # Use raw content if config allows it AND it's not too long, otherwise use snippet
            if self.include_raw_content and raw_content and len(raw_content) < 2000:
                used_content = raw_content
                content_type = "raw_content"
            elif content:
                used_content = content
                content_type = "content_snippet"
            else:
                continue
                
            if used_content:
                # Create a context entry with title
                if title:
                    formatted_content = f"**{title}**\n{used_content}"
                else:
                    formatted_content = used_content
                
                # Add source information
                chunk_header = f"\n--- Source {i}: {url} ---\n"
                formatted_chunk = f"{chunk_header}{formatted_content}"
                context_chunks.append(formatted_chunk)
                total_content_length += len(used_content)
                
                logger.info(f"LLM input chunk {i}: URL={url}, {content_type}_length={len(used_content)}")
        
        # Combine all context
        all_context = "\n\n".join(context_chunks)
        
        # Format the final prompt for LLM
        llm_prompt = f"""Answer the question based on the search results provided below. The search results contain relevant information from various web sources.

Search Results:
{all_context}

Question: {query}

Please provide a comprehensive answer based on the search results above."""
        
        # Log final output statistics
        output_length = len(llm_prompt)
        estimated_tokens = output_length // 4  # Rough estimate: 1 token ≈ 4 characters
        
        logger.info(f"LLM wrapper completed:")
        logger.info(f"  - Total context length: {len(all_context)} characters")
        logger.info(f"  - Final output length: {output_length} characters")
        logger.info(f"  - Estimated tokens: {estimated_tokens}")
        logger.info(f"  - Number of sources: {len(context_chunks)}")
        
        return llm_prompt
    
    async def close(self):
        """Close the client connection."""
        if hasattr(self.client, 'close'):
            await self.client.close()


def tavily_search(query: str) -> list[tuple[str, str]]:
    """
    Performs a search using the Tavily API with parameters optimized for high-quality results,
    advanced search depth, and fetching as much information as possible from each page.
    Prioritizes full raw content in markdown format when available, falling back to concatenated chunks.
    Assumes the TAVILY_API_KEY is set as an environment variable.
    
    :param query: The search query string.
    :return: A list of tuples, each containing (content_chunk, url).
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set.")
    
    client = TavilyClient(api_key=api_key)
    
    # Perform the search with parameters for high quality and maximum content
    response = client.search(
        query=query,
        search_depth="advanced",          # For more relevant and detailed sources
        max_results=10,                   # Reasonable number of high-quality results
        include_raw_content=True,         # Fixed: should be boolean, not string
        chunks_per_source=5,              # Reduced to default value
        include_answer=False,             # No need for generated answer
        include_images=False              # Focus on text content
    )
    
    results = response.get('results', [])
    output = []
    
    for result in results:
        # Prefer raw_content for full page info; fallback to content (which may be concatenated chunks)
        chunk = result.get('raw_content') or result.get('content', '')
        url = result.get('url', '')
        if chunk and url:
            output.append((chunk, url))
    
    # Note: Extract fallback disabled to prevent token overflow issues
    # The raw_content from search should be sufficient with proper chunk limits
    
    return output

if __name__ == "__main__":
    async def main():
        """
        Asynchronous example function demonstrating the usage of TavilySearcher for performing web searches.
        This function initializes a TavilySearcher instance with configurable parameters, executes a search query,
        and displays the results. It handles configuration and import errors gracefully, providing guidance for setup.
        Workflow:
            1. Initializes TavilySearcher with default or user-provided parameters.
            2. Executes a sample search query ("What are the recent news on Geoff hinton").
            3. Prints the number of results found and displays details (title, URL, content snippet) for up to 3 results.
            4. Handles errors such as missing API key or missing dependencies, printing helpful instructions.
            5. Closes the TavilySearcher client after completion.
        Exceptions:
            ValueError: Raised if configuration is invalid (e.g., missing API key).
            ImportError: Raised if required dependencies are not installed.
        Usage:
            - Ensure 'tavily-python' is installed.
            - Obtain an API key from https://tavily.com.
            - Set the TAVILY_API_KEY environment variable or pass the API key directly.
        Note:
            This function is intended as an example and should be run within an asynchronous event loop.
        """
        # Usage example
        print("TavilySearcher Usage Example")
        print("=" * 40)
        
        # Initialize with default values
        try:
            searcher = TavilySearcher(
                # api_key="your-api-key-here",  # Or set TAVILY_API_KEY env var
                max_results=10,
                include_raw_content=True,
                search_depth="advanced"
            )
            
            # Example search query - shorter version to avoid 400 char limit
            query = "factorization machines machine learning beginner explanation"
            print(f"Searching for: {query}")
            print("-" * 40)
            
            # Perform search
            results = await searcher.search(query)
            
            if "error" in results:
                print(f"Error: {results['error']}")
            else:
                print(f"Found {len(results.get('results', []))} results:")
                for i, result in enumerate(results.get('results', [])[:3]):
                    print(f"\n{i+1}. {result.get('title', 'No title')}")
                    print(f"   URL: {result.get('url', 'No URL')}")
                    print(f"   Content: {result.get('content', 'No content')[:1500]}...")
            
            # Demonstrate search_and_extract method
            print("\n" + "=" * 50)
            print("Testing search_and_extract method:")
            print("=" * 50)
            
            extracted_results = await searcher.search_and_extract(
                query="machine learning basics", 
                max_results=3,
                save_to_json="search_results"
            )
            
            print(f"Extracted content from {len(extracted_results)} pages:")
            for i, (content, url) in enumerate(extracted_results[:2]):
                print(f"\n{i+1}. URL: {url}")
                print(f"   Content length: {len(content)} characters")
                print(f"   Content preview: {content[:300]}...")
            
            # Demonstrate search_wrapper_for_llm_input method
            print("\n" + "=" * 50)
            print("Testing search_wrapper_for_llm_input method:")
            print("=" * 50)
            
            llm_formatted_result = await searcher.search_wrapper_for_llm_input(
                query="What are the benefits of machine learning?", 
                max_results=2
            )
            
            print(f"LLM-formatted result length: {len(llm_formatted_result)} characters")
            print(f"LLM-formatted result preview:\n{llm_formatted_result[:500]}...")
            
            # Close the client
            await searcher.close()
            
        except ValueError as e:
            print(f"Configuration Error: {e}")
            print("\nTo use this example:")
            print("1. Install tavily: pip install tavily-python")
            print("2. Get API key from https://tavily.com")
            print("3. Set environment variable: export TAVILY_API_KEY='your-key'")
            print("4. Or modify the code to pass api_key directly")
        except ImportError as e:
            print(f"Import Error: {e}")
    
    # Run the example
    asyncio.run(main())