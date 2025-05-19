from typing import List, Dict, Any
from langchain.schema import Document
from .base_agent import BaseAgent
from rag.retriever import HybridRetriever
from utils.pdf_parser import PDFParser

class ResearchAgent(BaseAgent):
    """Agent responsible for retrieving and processing academic papers."""
    
    def __init__(self, retriever: HybridRetriever, pdf_parser: PDFParser):
        super().__init__(
            name="ResearchAgent",
            description="Retrieves and processes academic papers"
        )
        self.retriever = retriever
        self.pdf_parser = pdf_parser
    
    async def process(self, query: str) -> List[Document]:
        """Process a research query.
        
        Args:
            query: Research query to process
            
        Returns:
            List of relevant documents
        """
        try:
            # Retrieve relevant documents
            documents = await self.retriever.retrieve(query)
            
            # Filter by relevance
            filtered_docs = await self.filter_by_relevance(documents)
            
            # Extract metadata
            for doc in filtered_docs:
                doc.metadata.update(await self.extract_metadata(doc))
            
            # Update agent state
            self.update_state(context={"documents": filtered_docs})
            
            return filtered_docs
            
        except Exception as e:
            self.add_error(f"Error processing query: {str(e)}")
            return []
    
    async def filter_by_relevance(self, documents: List[Document]) -> List[Document]:
        """Filter documents by relevance score.
        
        Args:
            documents: List of documents to filter
            
        Returns:
            Filtered list of documents
        """
        try:
            # Filter documents with relevance score > 0.5
            filtered_docs = [
                doc for doc in documents
                if doc.metadata.get("relevance_score", 0) > 0.5
            ]
            return filtered_docs
            
        except Exception as e:
            self.add_error(f"Error filtering documents: {str(e)}")
            return documents
    
    async def extract_metadata(self, doc: Document) -> Dict[str, Any]:
        """Extract metadata from document.
        
        Args:
            doc: Document to extract metadata from
            
        Returns:
            Dictionary of metadata
        """
        try:
            # Extract basic metadata
            metadata = {
                "title": doc.metadata.get("title", ""),
                "authors": doc.metadata.get("authors", []),
                "year": doc.metadata.get("year", ""),
                "source": doc.metadata.get("source", ""),
                "url": doc.metadata.get("url", ""),
                "relevance_score": doc.metadata.get("relevance_score", 0)
            }
            
            # Extract additional metadata if available
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                metadata.update(doc.metadata)
            
            return metadata
            
        except Exception as e:
            self.add_error(f"Error extracting metadata: {str(e)}")
            return {} 