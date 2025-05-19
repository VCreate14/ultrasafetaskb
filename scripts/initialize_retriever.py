import os
import asyncio
from dotenv import load_dotenv
from langchain.schema import Document
from rag.embeddings import EmbeddingsManager
from rag.retriever import HybridRetriever

async def initialize_retriever():
    """Initialize the retriever with sample documents."""
    # Load environment variables
    load_dotenv()
    
    # Initialize components
    embeddings_manager = EmbeddingsManager()
    retriever = HybridRetriever(embeddings_manager)
    
    # Read documents
    documents = []
    docs_dir = "data/documents"
    
    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(docs_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
                # Extract metadata from content
                lines = content.split("\n")
                metadata = {}
                
                for line in lines[:5]:  # First 5 lines contain metadata
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().lower()
                        value = value.strip()
                        
                        if key == "authors":
                            value = [author.strip() for author in value.split(",")]
                        elif key == "year":
                            value = int(value)
                            
                        metadata[key] = value
                
                # Create document
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
    
    # Initialize retriever
    await retriever.initialize(documents)
    print(f"Initialized retriever with {len(documents)} documents")

if __name__ == "__main__":
    asyncio.run(initialize_retriever()) 