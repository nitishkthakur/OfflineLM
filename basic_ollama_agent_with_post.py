import requests
import json
import inspect
from typing import List, Callable, Optional, Any, Dict, Generator, AsyncGenerator
from pydantic import BaseModel


class OllamaAgent:
    """
    A simple agent class for interacting with Ollama models with tool support and structured output.
    """
    
    def __init__(
        self,
        model_name: str,
        tools: List[Callable],
        output_schema: Optional[BaseModel] = None,
        endpoint: str = "http://localhost:11434/api/chat",
        proxies: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the Ollama Agent.
        
        Args:
            model_name: Name of the Ollama model to use
            tools: List of callable functions that can be used as tools
            output_schema: Optional Pydantic model for structured JSON output
            endpoint: URL of the Ollama chat API
            proxies: Proxy dictionary passed to ``requests.post``
        """
        self.model_name = model_name
        self.tools = tools
        self.output_schema = output_schema
        self.endpoint = endpoint
        self.proxies = proxies if proxies is not None else {"http": "", "https": ""}
        self.tool_schemas = self._generate_tool_schemas()
    
    def _generate_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Generate tool schemas from the provided functions based on their docstrings and signatures.
        
        Returns:
            List of tool schema dictionaries
        """
        schemas = []
        
        for tool in self.tools:
            # Get function signature
            sig = inspect.signature(tool)
            
            # Parse docstring for description and parameter info
            docstring = inspect.getdoc(tool) or ""
            
            # Basic schema structure
            schema = {
                "type": "function",
                "function": {
                    "name": tool.__name__,
                    "description": docstring.split('\n')[0] if docstring else f"Function {tool.__name__}",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            
            # Add parameters from function signature
            for param_name, param in sig.parameters.items():
                param_type = "string"  # Default type
                
                # Try to infer type from annotation
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == list:
                        param_type = "array"
                
                schema["function"]["parameters"]["properties"][param_name] = {
                    "type": param_type,
                    "description": f"Parameter {param_name}"
                }
                
                # Add to required if no default value
                if param.default == inspect.Parameter.empty:
                    schema["function"]["parameters"]["required"].append(param_name)
            
            schemas.append(schema)
        
        return schemas
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool by name with given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments to pass to the tool
            
        Returns:
            Result of the tool execution
        """
        for tool in self.tools:
            if tool.__name__ == tool_name:
                try:
                    return tool(**arguments)
                except Exception as e:
                    return f"Error executing {tool_name}: {str(e)}"
        
        return f"Tool {tool_name} not found"
    
    def invoke(self, prompt: str) -> Dict[str, Any]:
        """
        Send a prompt to the Ollama model and handle tool calls and structured output.
        
        Args:
            prompt: The input prompt to send to the model
            
        Returns:
            Dictionary containing the response and any tool results
        """
        try:
            # Prepare the request
            request_params = {
                'model': self.model_name,
                'messages': [{'role': 'user', 'content': prompt}],
                'stream': False
            }
            
            # Add tools if available
            if self.tool_schemas:
                request_params['tools'] = self.tool_schemas
            
            # Add format for structured output if schema is provided
            if self.output_schema:
                request_params['format'] = self.output_schema.model_json_schema()
            
            # Make the request to Ollama
            response = requests.post(
                self.endpoint,
                json=request_params,
                proxies=self.proxies,
            )
            response.raise_for_status()
            response = response.json()
            
            result = {
                'message': response['message']['content'],
                'tool_calls': [],
                'structured_output': None
            }
            
            # Handle tool calls if present
            if 'tool_calls' in response['message']:
                for tool_call in response['message']['tool_calls']:
                    tool_name = tool_call['function']['name']
                    tool_args = tool_call['function']['arguments']
                    
                    # Execute the tool
                    tool_result = self._execute_tool(tool_name, tool_args)
                    
                    result['tool_calls'].append({
                        'tool': tool_name,
                        'arguments': tool_args,
                        'result': tool_result
                    })
            
            # Handle structured output if schema is provided
            if self.output_schema and result['message']:
                try:
                    parsed_output = json.loads(result['message'])
                    result['structured_output'] = self.output_schema(**parsed_output)
                except (json.JSONDecodeError, ValueError) as e:
                    result['structured_output'] = f"Failed to parse structured output: {str(e)}"
            
            return result
            
        except Exception as e:
            return {
                'error': f"Failed to invoke model: {str(e)}",
                'message': None,
                'tool_calls': [],
                'structured_output': None
            }
        
    def invoke_plus_next_call(self, first_prompt: str, second_prompt: str, overall_task_prompt: str) -> Dict[str, Any]:
        """
        Perform a two-step invoke process where the output of the first call is used as input for the second call.
        
        Args:
            first_prompt: The initial prompt to send to the first invoke call
            second_prompt: The prompt to send in the second invoke call
            overall_task_prompt: The overall task context that frames the entire conversation
            
        Returns:
            Dictionary containing the final response and results from both calls
        """
        try:
            # First invoke call (with tools enabled)
            first_result = self.invoke(first_prompt)
            
            if 'error' in first_result:
                return {
                    'error': f"First invoke call failed: {first_result['error']}",
                    'first_result': first_result,
                    'second_result': None
                }
            
            # Get the output from the first call
            first_output = first_result.get('tool_calls', '')
            
            # Construct the input for the second invoke call
            second_invoke_input = (
                f"{overall_task_prompt}\n"
                f"<user>{first_prompt}</user>\n"
                f"<assistant>Output of first LLM Call: {first_output}</assistant>\n"
                f"<user>{second_prompt}</user>"
            )
            
            # Second invoke call (without tools)
            # Temporarily disable tools for the second call
            original_tool_schemas = self.tool_schemas
            self.tool_schemas = []
            
            try:
                second_result = self.invoke(second_invoke_input)
            finally:
                # Restore original tool schemas
                self.tool_schemas = original_tool_schemas
            
            return {
                'first_result': first_result,
                'second_result': second_result,
                'combined_input': second_invoke_input,
                'final_message': second_result.get('message') if 'error' not in second_result else second_result.get('error')
            }
            
        except Exception as e:
            return {
                'error': f"Failed in invoke_plus_next_call: {str(e)}",
                'first_result': None,
                'second_result': None
            }

class OllamaChat:
    """
    A simple chat class for interacting with Ollama models with conversation history.
    """
    
    def __init__(
        self,
        model: str = "gemma3:4b-it-fp16",
        endpoint: str = "http://localhost:11434/api/chat",
        proxies: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the Ollama Chat.
        
        Args:
            model: Name of the Ollama model to use
            endpoint: URL of the Ollama chat API
            proxies: Proxy dictionary passed to requests.post
        """
        self.model = model
        self.endpoint = endpoint
        self.proxies = proxies if proxies is not None else {"http": "", "https": ""}
        self.conversation_history = []

    def chat(self, prompt: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Send a message to the model and get a response while maintaining conversation history.
        
        Args:
            prompt: The user's message
            
        Returns:
            The model's response as a string
        """
        try:
            if conversation_history is not None:
                self.conversation_history = conversation_history

            # Add user message to history
            self.conversation_history.append({'role': 'user', 'content': prompt})
            
            # Prepare the request with full conversation history
            request_params = {
                'model': self.model,
                'messages': self.conversation_history,
                'stream': False,
                'keep_alive': '30m'
            }
            
            # Make the request to Ollama
            response = requests.post(
                self.endpoint,
                json=request_params,
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Extract the assistant's response
            assistant_message = response_data['message']['content']
            
            # Add assistant response to history
            self.conversation_history.append({'role': 'assistant', 'content': assistant_message})
            
            return assistant_message
            
        except Exception as e:
            error_msg = f"Error in chat: {str(e)}"
            # Don't add error to conversation history
            return error_msg

    def chat_stream(self, prompt: str, conversation_history: List[Dict[str, str]] = None) -> Generator[str, None, None]:
        """
        Send a message to the model and get a streaming response while maintaining conversation history.
        
        Args:
            prompt: The user's message
            
        Yields:
            Chunks of the model's response as they are generated
        """
        try:
            if conversation_history is not None:
                self.conversation_history = conversation_history

            # Add user message to history
            self.conversation_history.append({'role': 'user', 'content': prompt})
            
            # Prepare the request with full conversation history for streaming
            request_params = {
                'model': self.model,
                'messages': self.conversation_history,
                'stream': True,
                'keep_alive': '30m'
            }
            
            # Make the streaming request to Ollama
            response = requests.post(
                self.endpoint,
                json=request_params,
                stream=True
            )
            response.raise_for_status()
            
            full_response = ""
            
            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    try:
                        chunk_data = json.loads(line)
                        
                        # Check if this chunk contains content
                        if 'message' in chunk_data and 'content' in chunk_data['message']:
                            content_chunk = chunk_data['message']['content']
                            full_response += content_chunk
                            yield content_chunk
                        
                        # Check if this is the final chunk
                        if chunk_data.get('done', False):
                            break
                            
                    except json.JSONDecodeError:
                        # Skip malformed JSON lines
                        continue
            
            # Add the complete assistant response to history
            if full_response:
                self.conversation_history.append({'role': 'assistant', 'content': full_response})
            
        except Exception as e:
            error_msg = f"Error in streaming chat: {str(e)}"
            yield error_msg
    
    async def chat_stream_async(self, prompt: str, conversation_history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
        """
        Async version of chat_stream for use with async web frameworks like FastAPI.
        
        Args:
            prompt: The user's message
            
        Yields:
            Chunks of the model's response as they are generated
        """
        import asyncio
        import aiohttp
        
        try:
            if conversation_history is not None:
                self.conversation_history = conversation_history

            # Add user message to history
            self.conversation_history.append({'role': 'user', 'content': prompt})
            
            # Prepare the request with full conversation history for streaming
            request_params = {
                'model': self.model,
                'messages': self.conversation_history,
                'stream': True,
                'keep_alive': '30m'
            }
            
            full_response = ""
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.endpoint, json=request_params) as response:
                    response.raise_for_status()
                    
                    async for line in response.content:
                        if line:
                            line = line.decode('utf-8').strip()
                            if line:
                                try:
                                    chunk_data = json.loads(line)
                                    
                                    # Check if this chunk contains content
                                    if 'message' in chunk_data and 'content' in chunk_data['message']:
                                        content_chunk = chunk_data['message']['content']
                                        full_response += content_chunk
                                        yield content_chunk
                                    
                                    # Check if this is the final chunk
                                    if chunk_data.get('done', False):
                                        break
                                        
                                except json.JSONDecodeError:
                                    # Skip malformed JSON lines
                                    continue
            
            # Add the complete assistant response to history
            if full_response:
                self.conversation_history.append({'role': 'assistant', 'content': full_response})
            
        except Exception as e:
            error_msg = f"Error in async streaming chat: {str(e)}"
            yield error_msg

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()


class GroqChat:
    """
    A simple chat class for interacting with Groq models with conversation history.
    Mirrors OllamaChat functionality but uses Groq API.
    """
    
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Groq Chat.
        
        Args:
            model: Name of the Groq model to use
            api_key: Groq API key (if None, will try to get from environment)
        """
        try:
            from groq import Groq
            import os
            # Try to load .env file if python-dotenv is available
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass  # python-dotenv not available, continue without it
        except ImportError:
            raise ImportError("groq package is required. Install with: pip install groq")
        
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY or GROK_API_KEY must be provided or set in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        self.conversation_history = []

    def chat(self, prompt: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Send a message to the model and get a response while maintaining conversation history.
        
        Args:
            prompt: The user's message
            conversation_history: Optional conversation history to use instead of instance history
            
        Returns:
            The model's response as a string
        """
        try:
            if conversation_history is not None:
                self.conversation_history = conversation_history

            # Add user message to history
            self.conversation_history.append({'role': 'user', 'content': prompt})
            
            # Make the request to Groq
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=2048
            )
            
            # Extract the assistant's response
            assistant_message = completion.choices[0].message.content
            
            # Add assistant response to history
            self.conversation_history.append({'role': 'assistant', 'content': assistant_message})
            
            return assistant_message
            
        except Exception as e:
            error_msg = f"Error in Groq chat: {str(e)}"
            # Don't add error to conversation history
            return error_msg

    def chat_stream(self, prompt: str, conversation_history: List[Dict[str, str]] = None) -> Generator[str, None, None]:
        """
        Send a message to the model and get a streaming response while maintaining conversation history.
        
        Args:
            prompt: The user's message
            conversation_history: Optional conversation history to use instead of instance history
            
        Yields:
            Chunks of the model's response as they are generated
        """
        try:
            if conversation_history is not None:
                self.conversation_history = conversation_history

            # Add user message to history
            self.conversation_history.append({'role': 'user', 'content': prompt})
            
            # Make the streaming request to Groq
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=2048,
                stream=True
            )
            
            full_response = ""
            
            # Process the streaming response
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content_chunk = chunk.choices[0].delta.content
                    full_response += content_chunk
                    yield content_chunk
            
            # Add the complete assistant response to history
            if full_response:
                self.conversation_history.append({'role': 'assistant', 'content': full_response})
            
        except Exception as e:
            error_msg = f"Error in Groq streaming chat: {str(e)}"
            yield error_msg
    
    async def chat_stream_async(self, prompt: str, conversation_history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
        """
        Async version of chat_stream for use with async web frameworks like FastAPI.
        
        Args:
            prompt: The user's message
            conversation_history: Optional conversation history to use instead of instance history
            
        Yields:
            Chunks of the model's response as they are generated
        """
        import asyncio
        
        try:
            if conversation_history is not None:
                self.conversation_history = conversation_history

            # Add user message to history
            self.conversation_history.append({'role': 'user', 'content': prompt})
            
            # Run the synchronous streaming in a thread to make it async-compatible
            def _sync_stream():
                try:
                    stream = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.conversation_history,
                        temperature=0.7,
                        max_tokens=2048,
                        stream=True
                    )
                    
                    full_response = ""
                    chunks = []
                    
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            content_chunk = chunk.choices[0].delta.content
                            full_response += content_chunk
                            chunks.append(content_chunk)
                    
                    return chunks, full_response
                except Exception as e:
                    return [f"Error in Groq async streaming chat: {str(e)}"], ""
            
            # Run in thread pool to avoid blocking
            chunks, full_response = await asyncio.get_event_loop().run_in_executor(None, _sync_stream)
            
            # Yield chunks
            for chunk in chunks:
                yield chunk
            
            # Add the complete assistant response to history
            if full_response and not full_response.startswith("Error"):
                self.conversation_history.append({'role': 'assistant', 'content': full_response})
            
        except Exception as e:
            error_msg = f"Error in Groq async streaming chat: {str(e)}"
            yield error_msg

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()
    
    def start_new_chat(self):
        """Start a new chat by clearing the conversation history."""
        self.clear_history()

# Example usage:
if __name__ == "__main__":
    # Example tool functions
    def get_product(a: int, b: int) -> int:
        """
        Computes the product of two numbers.
        """
        return int(a) * int(b)

    def calculate_sum(a: int, b: int) -> int:
        """
        Calculate the sum of two numbers.
        
        Args:
            a: First number
            b: Second number
        """
        return a + b
    
    def calculate_weather(location: str) -> Dict[str, Any]:
        """
        Fetches weather information for a given location.
        
        Args:
            location: Name of the location to get weather for
            
        Returns:
            Dictionary with weather details
        """
        # Simulated response, replace with actual API call if needed
        if location.lower() == "india":
            temperature = 35
        else:
            temperature = 20
        return {
            "location": location,
            "temperature": temperature,
            "units": "Celsius",
            "description": "Sunny"
        }
    
    # Example output schema
    class WeatherResponse(BaseModel):
        location: str
        temperature: float
        units: str
        description: str
    
    # Create agent
    agent = OllamaAgent(
        model_name="qwen2.5:7b",
        tools=[get_product, calculate_sum, calculate_weather],
        output_schema=None
    )
    
    # Use agent - on one call
    result = agent.invoke("What's the Sum of 11 and 22? Also, what's the product of 11 and 22?")
    #print(result)
    
    # Use agent - on two step process
    result = agent.invoke_plus_next_call(first_prompt = "What's the Sum of 11 and 22? Also, what's the product of 11 and 26? and let me know the weather in india",
                                         second_prompt="Now, write the final answer to the user questions based on the above conversation",
                                         overall_task_prompt="You are a helpful assistant that provides answers based on user queries based on only the conversation to follow. If any information you need is not present in the following conversation, you mention so")
    print(result)
    
    # Example usage of chat
    chat_agent = OllamaChat(model="qwen2.5:7b")
    
    # Example of regular chat
    print("=== Regular Chat ===")
    chat_result = chat_agent.chat("Hello, who won the world series in 2020?")
    print(chat_result)
    
    # Example of streaming chat
    print("\n=== Streaming Chat ===")
    print("Question: What is artificial intelligence and how does it work?")
    print("Streaming Response: ", end="", flush=True)
    for chunk in chat_agent.chat_stream("What is artificial intelligence and how does it work?"):
        print(chunk, end="", flush=True)
    print()  # New line after streaming is complete
    
    # Print conversation history
    print("\nConversation History:")
    for message in chat_agent.get_history():
        print(f"{message['role']}: {message['content'][:100]}...")  # Truncate for readability
    
    # Clear history
    chat_agent.clear_history()


