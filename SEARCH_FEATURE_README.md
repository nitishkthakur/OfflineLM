# Search Integration Feature

The OfflineLM chat application now includes integrated web search functionality powered by Tavily Search API.

## Features

- **Web Search Integration**: Automatically search the web before generating responses
- **Configurable Search**: Toggle search on/off and set the number of search results (1-20)
- **Real-time Feedback**: See search progress with "Searching (n sites) and thinking..." messages
- **Seamless Integration**: Search results are passed to the selected LLM model for context-aware responses

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements_streaming.txt
```

The following packages are included:
- `tavily-python>=0.4.0`: Tavily search client
- `python-dotenv>=1.0.0`: Environment variable management

### 2. Get Tavily API Key

1. Sign up at [Tavily](https://app.tavily.com/)
2. Get your API key from the dashboard
3. Create a `.env` file in the project root:

```bash
# .env
TAVILY_API_KEY=your_tavily_api_key_here
```

Or set as environment variable:
```bash
export TAVILY_API_KEY='your_tavily_api_key_here'
```

### 3. Start the Server

```bash
python fastapi_streaming_improved.py
```

Open http://localhost:8001 in your browser.

## Usage

### Web Interface

1. **Enable Search**: Toggle the "Enable Search" switch in the left sidebar
2. **Set Search Count**: Use the number input to set how many sites to search (default: 5)
3. **Ask Questions**: Type your question and send - search will happen automatically if enabled

### Search Process

When search is enabled, the following happens:

1. **User sends message** → Search toggle is checked
2. **Search initiated** → "Searching (n sites) and thinking..." message appears
3. **Web search executed** → Tavily searches the web for relevant content
4. **Results processed** → Search results are formatted for LLM consumption
5. **LLM response** → Selected model generates response using search context
6. **Response streamed** → Answer is streamed back to the user

### Visual Indicators

- **Loading Message**: Shows "Searching (5 sites) and thinking..." during search
- **Status Updates**: Status bar shows search progress
- **Error Handling**: Graceful fallback if search fails

## Technical Implementation

### Backend Changes

1. **TavilySearcher Integration** (`fastapi_streaming_improved.py`):
   - Added search parameters to streaming endpoint
   - Integrated `search_wrapper_for_llm_input` method
   - Added search status messages

2. **Search Parameters**:
   - `search_enabled`: Boolean to enable/disable search
   - `search_count`: Number of sites to search (1-20)

### Frontend Changes

1. **UI Controls** (`minimal_ui_streaming.html`):
   - Search toggle switch
   - Search count number input
   - Dynamic loading messages

2. **JavaScript Updates** (`script.js`):
   - Pass search parameters to backend
   - Handle search status messages
   - Update loading indicators

### API Endpoint Updates

The main streaming endpoint now accepts additional parameters:

```
GET /chat/stream-sse
Parameters:
  - message: str (user question)
  - model: str (selected model)
  - session: str (session ID)
  - search_enabled: bool (enable search)
  - search_count: int (number of results)
```

## Testing

Run the integration test:

```bash
python test_search_integration.py
```

This will test the search functionality without needing the full server.

## Error Handling

The system gracefully handles various error scenarios:

- **No API Key**: Search is skipped, normal chat continues
- **Search API Errors**: Fallback to normal chat with error message
- **Network Issues**: Timeout and retry with fallback
- **Invalid Parameters**: Uses defaults (5 results)

## Configuration

### Environment Variables

- `TAVILY_API_KEY`: Your Tavily API key
- `GROQ_API_KEY`: Your Groq API key (for Groq models)

### Search Settings

- **Max Results**: 1-20 (default: 5)
- **Search Depth**: Advanced (configured in TavilySearcher)
- **Include Raw Content**: True (for comprehensive context)

## Limitations

- Requires active internet connection for search
- Search results depend on Tavily's web index
- API rate limits may apply based on your Tavily plan
- Search adds latency to response generation

## Troubleshooting

### Common Issues

1. **Search not working**:
   - Check TAVILY_API_KEY is set correctly
   - Verify internet connection
   - Check Tavily service status

2. **"Search failed" message**:
   - API key might be invalid or expired
   - Rate limit exceeded
   - Temporary service issue

3. **Toggle not visible**:
   - Clear browser cache
   - Check JavaScript console for errors

### Debug Mode

Enable debug logging by checking the browser console and server logs for detailed error information.

## Future Enhancements

Potential improvements for future versions:

- **Search Source Display**: Show which websites were searched
- **Search History**: Remember recent search queries
- **Advanced Filters**: Filter by domain, date, content type
- **Caching**: Cache search results for repeated queries
- **Search Analytics**: Track search usage and effectiveness