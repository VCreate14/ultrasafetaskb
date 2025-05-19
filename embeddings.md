# Embeddings Documentation

## Overview
The Embeddings Manager component handles text embedding generation using sentence-transformers. It provides a simple interface for converting text into vector representations that can be used for semantic search.

## Features

- Text to vector conversion
- Batch processing support
- Configurable model selection
- Efficient memory usage

## Usage

### Basic Usage

```python
from rag import EmbeddingsManager

# Initialize with default model
embeddings_manager = EmbeddingsManager()

# Get embedding for single text
embedding = embeddings_manager.get_embedding("Your text here")

# Get embeddings for multiple texts
embeddings = embeddings_manager.get_embeddings([
    "First text",
    "Second text",
    "Third text"
])
```

### Custom Model

```python
# Initialize with custom model
embeddings_manager = EmbeddingsManager(
    model_name="your-preferred-model"
)
```

## API Reference

### EmbeddingsManager

```python
class EmbeddingsManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2")
    """
    Initialize the embeddings manager.
    
    Args:
        model_name: Name of the sentence-transformer model to use
    """
    
    def get_embedding(self, text: str) -> List[float]
    """
    Get embedding for a single text.
    
    Args:
        text: Input text to embed
        
    Returns:
        List of floats representing the text embedding
    """
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]
    """
    Get embeddings for multiple texts.
    
    Args:
        texts: List of input texts to embed
        
    Returns:
        List of embeddings, one for each input text
    """
```

## Supported Models

The system supports any model from the sentence-transformers library. Default model is `all-MiniLM-L6-v2`.

Popular alternatives:
- `all-mpnet-base-v2`
- `all-MiniLM-L12-v2`
- `multi-qa-mpnet-base-dot-v1`

## Best Practices

1. **Batch Processing**
   - Use `get_embeddings()` for multiple texts
   - Optimal batch size depends on available memory
   - Consider your GPU memory if using CUDA

2. **Model Selection**
   - Choose model based on your needs:
     - Speed vs. accuracy
     - Language support
     - Vector dimensions
   - Consider memory requirements

3. **Performance**
   - Cache frequently used embeddings
   - Use appropriate batch sizes
   - Monitor memory usage

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| EMBEDDINGS_MODEL | Model name | all-MiniLM-L6-v2 |

### Model Properties

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| all-MiniLM-L6-v2 | 384 | Fast | Good |
| all-mpnet-base-v2 | 768 | Medium | Better |
| all-MiniLM-L12-v2 | 384 | Medium | Better |

## Limitations

1. **Memory Usage**
   - Models require significant memory
   - Batch size affects memory usage
   - Consider GPU memory if available

2. **Performance**
   - First run loads model into memory
   - Batch processing more efficient
   - CPU vs GPU performance varies

3. **Model Limitations**
   - Language support varies by model
   - Quality depends on training data
   - May not handle all text types well

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size
   - Use smaller model
   - Clear memory between batches

2. **Performance Issues**
   - Use GPU if available
   - Optimize batch size
   - Consider model size

3. **Quality Issues**
   - Try different model
   - Preprocess text
   - Check input format

## Examples

### Text Similarity

```python
from rag import EmbeddingsManager
import numpy as np

# Initialize
embeddings_manager = EmbeddingsManager()

# Get embeddings
text1 = "The quick brown fox"
text2 = "A fast brown fox"
text3 = "The weather is nice"

emb1 = embeddings_manager.get_embedding(text1)
emb2 = embeddings_manager.get_embedding(text2)
emb3 = embeddings_manager.get_embedding(text3)

# Calculate similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Should be high
similarity1_2 = cosine_similarity(emb1, emb2)

# Should be low
similarity1_3 = cosine_similarity(emb1, emb3)
```

### Batch Processing

```python
# Process multiple texts efficiently
texts = [
    "First document",
    "Second document",
    "Third document",
    # ... more texts
]

# Get all embeddings at once
embeddings = embeddings_manager.get_embeddings(texts)
```

## Contributing

To contribute to the embeddings component:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This component is part of the RAG system and follows the same licensing terms. 