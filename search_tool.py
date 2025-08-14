import asyncio
import os
from typing import Optional, List, Dict, Any
try:
    from tavily import AsyncTavilyClient

except ImportError:
    print("tavily-python not installed. Install with: pip install tavily-python")
    AsyncTavilyClient = None

from os import getenv
from dotenv import load_dotenv

load_dotenv()

class TavilySearcher:
    """
    A wrapper class for Tavily search functionality with configurable defaults.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        max_results: int = 5,
        include_raw_content: str = "markdown",
        search_depth: str = "advanced"
    ):
        """
        Initialize TavilySearcher with default parameters.
        
        Args:
            api_key: Tavily API key. If None, will try to get from TAVILY_API_KEY env var
            max_results: Maximum number of search results (default: 5)
            include_raw_content: Format for raw content (default: "markdown")
            search_depth: Search depth level (default: "advanced")
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
    
    async def search(
        self, 
        query: str,
        max_results: Optional[int] = None,
        include_raw_content: Optional[str] = None,
        search_depth: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform an asynchronous search using Tavily.
        
        Args:
            query: Search query string
            max_results: Override default max_results
            include_raw_content: Override default include_raw_content
            search_depth: Override default search_depth
            **kwargs: Additional parameters to pass to Tavily search
            
        Returns:
            Dictionary containing search results
        """
        search_params = {
            "query": query,
            "max_results": max_results or self.max_results,
            "search_depth": search_depth or self.search_depth,
            **kwargs
        }
        
        # Add optional parameters if they have values
        if include_raw_content or self.include_raw_content:
            search_params["include_raw_content"] = include_raw_content or self.include_raw_content
        
        try:
            results = await self.client.search(**search_params)
            return results
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}
    
    async def extract(self, urls: List[str], **kwargs) -> Dict[str, Any]:
        """Extract content from URLs."""
        try:
            return await self.client.extract(urls, **kwargs)
        except Exception as e:
            return {"error": f"Extract failed: {str(e)}"}
    
    async def close(self):
        """Close the client connection."""
        if hasattr(self.client, 'close'):
            await self.client.close()


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
                max_results=20,
                include_raw_content="markdown",
                search_depth="advanced"
            )
            
            # Example search query
            query = "what are factorization machines - refer to papers. Explain it to me like I am a beginner but include all details"
            print(f"Searching for: {query}")
            print("-" * 40)
            
            # Perform search
            results = await searcher.search(query)
            
            if "error" in results:
                print(f"Error: {results['error']}")
            else:
                print(f"Found {len(results.get('results', []))} results:")
                for i, result in enumerate(results.get('results', [])[:5]):
                    print(f"\n{i+1}. {result.get('title', 'No title')}")
                    print(f"   URL: {result.get('url', 'No URL')}")
                    print(f"   Content: {result.get('content', 'No content')[:2000]}...")
            
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