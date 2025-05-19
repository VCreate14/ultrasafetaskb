from typing import List, Dict, Any
from langchain.schema import Document
from qdrant_client import QdrantClient
from .embeddings import EmbeddingsManager
from .web_search import WebSearch
import os
from dotenv import load_dotenv

class HybridRetriever:
    """Combines keyword and semantic search for document retrieval."""
    
    def __init__(self, embeddings_manager: EmbeddingsManager):
        """Initialize the retriever.
        
        Args:
            embeddings_manager: Embeddings manager instance
        """
        load_dotenv()
        self.embeddings_manager = embeddings_manager
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.collection_name = "academic_papers"
        self.web_search = WebSearch()
    
    async def initialize(self, documents: List[Document]) -> None:
        """Initialize the retriever with documents.
        
        Args:
            documents: List of documents to index
        """
        try:
            # Generate embeddings
            texts = [doc.page_content for doc in documents]
            embeddings = self.embeddings_manager.get_embeddings(texts)
            
            # Prepare points for Qdrant
            points = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                points.append({
                    "id": i,
                    "vector": embedding,
                    "payload": {
                        "text": doc.page_content,
                        "metadata": doc.metadata
                    }
                })
            
            # Create collection if it doesn't exist
            collections = self.qdrant_client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "size": len(embeddings[0]),
                        "distance": "Cosine"
                    }
                )
            
            # Upload points
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
        except Exception as e:
            raise Exception(f"Error initializing retriever: {str(e)}")
    
    async def retrieve(self, query: str, limit: int = 10, include_web: bool = True) -> List[Document]:
        """Retrieve relevant documents.
        
        Args:
            query: Search query
            limit: Maximum number of documents to retrieve
            include_web: Whether to include web search results
            
        Returns:
            List of relevant documents
        """
        try:
            documents = []
            
            # Get documents from Qdrant
            query_embedding = self.embeddings_manager.get_embedding(query)
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            # Convert Qdrant results to documents
            for result in search_result:
                doc = Document(
                    page_content=result.payload["text"],
                    metadata=result.payload["metadata"]
                )
                doc.metadata["relevance_score"] = result.score
                doc.metadata["source"] = "database"
                documents.append(doc)
            
            # Add web search results if requested
            if include_web:
                web_docs = self.web_search.search_and_extract(query, max_results=limit)
                documents.extend(web_docs)
            
            # Sort by relevance score if available
            documents.sort(key=lambda x: x.metadata.get("relevance_score", 0), reverse=True)
            
            return documents[:limit]
            
        except Exception as e:
            raise Exception(f"Error retrieving documents: {str(e)}")
    
    async def delete_collection(self) -> None:
        """Delete the collection."""
        try:
            self.qdrant_client.delete_collection(
                collection_name=self.collection_name
            )
        except Exception as e:
            raise Exception(f"Error deleting collection: {str(e)}") 