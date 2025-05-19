from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dotenv import load_dotenv

class EmbeddingsManager:
    """Manages text embeddings using Sentence Transformers."""
    
    def __init__(self, model_name: str = None):
        """Initialize the embeddings manager.
        
        Args:
            model_name: Name of the Sentence Transformer model to use.
                      Defaults to environment variable or 'all-MiniLM-L6-v2'
        """
        load_dotenv()
        self.model_name = model_name or os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_name)
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        try:
            # Generate embeddings
            embeddings = self.model.encode(texts)
            
            # Convert to list of lists
            return embeddings.tolist()
            
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector
        """
        try:
            # Generate embedding
            embedding = self.model.encode(text)
            
            # Convert to list
            return embedding.tolist()
            
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        try:
            # Generate embeddings
            embedding1 = self.model.encode(text1)
            embedding2 = self.model.encode(text2)
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            
            return float(similarity)
            
        except Exception as e:
            raise Exception(f"Error computing similarity: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "max_sequence_length": self.model.max_seq_length
        } 