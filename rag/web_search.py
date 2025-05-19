from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
import re
from urllib.parse import urlparse
import time

class WebSearch:
    """Web search and content retrieval using DuckDuckGo."""
    
    def __init__(self):
        """Initialize the web search component."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search the web using DuckDuckGo.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with title, link, and snippet
        """
        try:
            # Construct DuckDuckGo search URL
            search_url = f"https://html.duckduckgo.com/html/?q={query}"
            
            # Make request
            response = requests.get(search_url, headers=self.headers)
            response.raise_for_status()
            
            # Parse results
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Extract search results
            for result in soup.select('.result')[:max_results]:
                title_elem = result.select_one('.result__title')
                snippet_elem = result.select_one('.result__snippet')
                link_elem = result.select_one('.result__url')
                
                if title_elem and snippet_elem and link_elem:
                    title = title_elem.get_text(strip=True)
                    snippet = snippet_elem.get_text(strip=True)
                    link = link_elem.get_text(strip=True)
                    
                    results.append({
                        'title': title,
                        'snippet': snippet,
                        'link': link
                    })
            
            return results
            
        except Exception as e:
            raise Exception(f"Error searching web: {str(e)}")
    
    def extract_content(self, url: str) -> str:
        """Extract main content from a webpage.
        
        Args:
            url: URL to extract content from
            
        Returns:
            Extracted text content
        """
        try:
            # Add scheme if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Make request
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Parse content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up text
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove multiple newlines
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            
            return text
            
        except Exception as e:
            raise Exception(f"Error extracting content: {str(e)}")
    
    def search_and_extract(self, query: str, max_results: int = 5) -> List[Document]:
        """Search the web and extract content from results.
        
        Args:
            query: Search query
            max_results: Maximum number of results to process
            
        Returns:
            List of documents with extracted content
        """
        try:
            # Search web
            results = self.search(query, max_results)
            documents = []
            
            # Extract content from each result
            for result in results:
                try:
                    content = self.extract_content(result['link'])
                    
                    # Create document
                    doc = Document(
                        page_content=content,
                        metadata={
                            'title': result['title'],
                            'url': result['link'],
                            'snippet': result['snippet'],
                            'source': 'web'
                        }
                    )
                    documents.append(doc)
                    
                    # Be nice to servers
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error processing {result['link']}: {str(e)}")
                    continue
            
            return documents
            
        except Exception as e:
            raise Exception(f"Error in search and extract: {str(e)}") 