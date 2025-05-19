# Web Search Documentation

## Overview
The web search component provides DuckDuckGo-based search capabilities integrated with the RAG system. It allows for web content retrieval and integration with the existing document retrieval system.

## Components

### WebSearch Class
The `WebSearch` class provides web search and content extraction capabilities.

#### Features
- DuckDuckGo web search
- Web page content extraction
- Content cleaning and formatting
- Rate limiting to be respectful to servers

#### Methods

##### `search(query: str, max_results: int = 5) -> List[Dict[str, str]]`
Performs a web search using DuckDuckGo.

Parameters:
- `query`: Search query string
- `max_results`: Maximum number of results to return (default: 5)

Returns:
- List of dictionaries containing:
  - `title`: Page title
  - `snippet`: Search result snippet
  - `link`: URL of the page

##### `extract_content(url: str) -> str`
Extracts main content from a webpage.

Parameters:
- `url`: URL to extract content from

Returns:
- Extracted text content

##### `search_and_extract(query: str, max_results: int = 5) -> List[Document]`
Combines search and content extraction into a single operation.

Parameters:
- `query`: Search query string
- `max_results`: Maximum number of results to process (default: 5)

Returns:
- List of LangChain Document objects with:
  - `page_content`: Extracted text
  - `metadata`: Contains title, URL, snippet, and source information

### HybridRetriever Integration

The `HybridRetriever` class has been enhanced to include web search capabilities.

#### New Features
- Combined database and web search results
- Relevance-based sorting
- Configurable web search inclusion

#### Usage

```python
from rag import HybridRetriever, EmbeddingsManager

# Initialize components
embeddings_manager = EmbeddingsManager()
retriever = HybridRetriever(embeddings_manager)

# Retrieve documents with web search
documents = await retriever.retrieve(
    query="your search query",
    limit=10,
    include_web=True  # Set to False for database-only results
)
```

## Example Usage

```python
from rag import WebSearch

# Initialize web search
web_search = WebSearch()

# Simple search
results = web_search.search("python programming")

# Extract content from a specific URL
content = web_search.extract_content("https://example.com")

# Combined search and extract
documents = web_search.search_and_extract("machine learning tutorials")
```

## Best Practices

1. **Rate Limiting**
   - The component includes built-in delays between requests
   - Respects server resources and avoids overwhelming websites

2. **Error Handling**
   - Gracefully handles network errors
   - Continues processing even if individual results fail

3. **Content Cleaning**
   - Removes unwanted elements (scripts, styles, etc.)
   - Normalizes whitespace and formatting
   - Preserves important content structure

## Dependencies

Required packages:
- `beautifulsoup4>=4.12.0`
- `requests>=2.31.0`

Install using:
```bash
pip install -r requirements.txt
```

## Limitations

1. **Content Extraction**
   - May not perfectly extract content from all websites
   - Some dynamic content may not be captured
   - JavaScript-rendered content is not supported

2. **Search Results**
   - Limited to DuckDuckGo's search capabilities
   - Results may vary based on region and time

3. **Rate Limiting**
   - Built-in delays may slow down bulk operations
   - Consider caching for frequently accessed content

## Troubleshooting

Common issues and solutions:

1. **No Results Found**
   - Check internet connection
   - Verify search query
   - Try different search terms

2. **Content Extraction Failures**
   - Verify URL accessibility
   - Check for website blocking
   - Try alternative URLs

3. **Performance Issues**
   - Reduce `max_results` parameter
   - Implement caching
   - Use database results when possible

## Contributing

To contribute to the web search component:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This component is part of the RAG system and follows the same licensing terms. 