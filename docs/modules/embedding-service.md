# EmbeddingService Module

## Overview

The `EmbeddingService` module handles text embedding generation using Ollama's nomic-embed-text model with batch processing, error recovery, dimension handling, and comprehensive service monitoring.

## Features

- **Ollama integration** with nomic-embed-text model
- **Batch processing** for improved performance
- **Error recovery** with fallback embeddings
- **Service monitoring** with health checks and statistics
- **Dimension validation** and consistency checks
- **Timeout handling** and connection management

## Installation

```bash
# Requires Ollama with nomic-embed-text model
ollama pull nomic-embed-text
pip install requests
```

## Basic Usage

### Single Text Embedding

```python
from modules.embedding_service import EmbeddingService

service = EmbeddingService()
text = "This is a sample text to embed"
embedding = service.embed_single(text)

print(f"Embedding dimension: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
```

### Batch Embedding

```python
from modules.embedding_service import EmbeddingService

service = EmbeddingService()
texts = [
    "First text to embed",
    "Second text to embed", 
    "Third text to embed"
]

embeddings = service.embed(texts)
print(f"Generated {len(embeddings)} embeddings")
```

### Custom Configuration

```python
service = EmbeddingService(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
    default_dimension=768,
    timeout=60,
    batch_size=20
)
```

## API Reference

### EmbeddingService Class

#### `__init__(model="nomic-embed-text", base_url="http://localhost:11434", default_dimension=768, timeout=30, batch_size=10)`

Initialize EmbeddingService with configuration.

**Parameters:**
- `model`: Ollama embedding model to use
- `base_url`: Ollama server URL  
- `default_dimension`: Default embedding dimension for fallback
- `timeout`: Request timeout in seconds
- `batch_size`: Number of texts to process in parallel

#### `embed(texts: List[str]) -> List[List[float]]`

Generate embeddings for a list of texts.

**Parameters:**
- `texts`: List of text strings to embed

**Returns:**
- List of embedding vectors (one per input text)

#### `embed_single(text: str) -> List[float]`

Generate embedding for a single text.

**Parameters:**
- `text`: Text string to embed

**Returns:**
- Embedding vector for the text

#### `test_connection() -> bool`

Test connection to Ollama embedding service.

**Returns:**
- True if service is available, False otherwise

#### `get_statistics() -> Dict[str, Any]`

Get service usage statistics.

**Returns:**
- Dictionary with service statistics

#### `health_check() -> Dict[str, Any]`

Perform comprehensive health check of the embedding service.

**Returns:**
- Health check results

## Service Statistics

The service tracks comprehensive usage statistics:

```python
service = EmbeddingService()
# ... process some texts ...

stats = service.get_statistics()
print(f"Total requests: {stats['total_requests']}")
print(f"Success rate: {stats['success_rate']:.1f}%")
print(f"Processing speed: {stats['texts_per_second']:.1f} texts/sec")
```

### Tracked Metrics

- **total_requests**: Number of API requests made
- **successful_requests**: Number of successful requests
- **failed_requests**: Number of failed requests
- **total_texts_processed**: Total texts processed
- **total_processing_time**: Total time spent processing
- **fallback_embeddings**: Number of fallback embeddings used
- **success_rate**: Calculated success percentage
- **texts_per_second**: Processing throughput

## Health Monitoring

### Basic Health Check

```python
service = EmbeddingService()
health = service.health_check()

print(f"Service available: {health['service_available']}")
print(f"Model loaded: {health['model_loaded']}")
print(f"Embedding test: {health['embedding_test']}")
print(f"Response time: {health['response_time']:.3f}s")
```

### Health Check Results

```python
{
    "service_available": True,
    "model_loaded": True,
    "embedding_test": True,
    "response_time": 0.145,
    "error": None
}
```

### Connection Testing

```python
service = EmbeddingService()

if service.test_connection():
    print("✅ Service is working correctly")
else:
    print("❌ Service test failed")
```

## Error Handling

### Graceful Degradation

When embedding generation fails, the service provides fallback embeddings:

```python
# Fallback embedding (zero vector)
fallback = service._get_fallback_embedding()
print(f"Fallback dimension: {len(fallback)}")  # 768 zeros
```

### Error Recovery

- **Network timeouts**: Automatic retry with fallback
- **Model unavailable**: Fallback embeddings returned
- **Invalid responses**: Error logging with fallback
- **Service overload**: Batch size auto-adjustment

### Error Reporting

```python
try:
    embeddings = service.embed(texts)
except Exception as e:
    print(f"Embedding failed: {e}")
    # Service continues with fallback embeddings
```

## Batch Processing

### Optimal Batch Sizes

- **Small batch (5-10)**: Lower memory usage, good for testing
- **Medium batch (10-20)**: Balanced performance (recommended)
- **Large batch (20+)**: Higher throughput, more memory usage

### Batch Processing Flow

1. **Input validation**: Check text list
2. **Batch splitting**: Divide into manageable chunks
3. **Parallel processing**: Process batches concurrently
4. **Error handling**: Individual text failures don't stop batch
5. **Statistics update**: Track processing metrics

## Performance Optimization

### Text Length Handling

```python
# Automatic text truncation for very long texts
service = EmbeddingService()
long_text = "..." * 10000  # Very long text
embedding = service.embed_single(long_text)  # Automatically truncated
```

### Dimension Validation

```python
# Automatic dimension validation
service = EmbeddingService(default_dimension=768)
embedding = service.embed_single("test")

if len(embedding) != 768:
    print("Warning: Unexpected embedding dimension")
```

### Performance Monitoring

```python
import time

start_time = time.time()
embeddings = service.embed(texts)
processing_time = time.time() - start_time

print(f"Processed {len(texts)} texts in {processing_time:.2f}s")
print(f"Rate: {len(texts)/processing_time:.1f} texts/second")
```

## Integration Examples

### With TextChunker

```python
from modules.text_chunker import TextChunker
from modules.embedding_service import EmbeddingService

# Create chunks
chunker = TextChunker()
chunks = chunker.create_enhanced_chunks(page_texts)

# Generate embeddings
service = EmbeddingService()
texts = [chunk.text for chunk in chunks]
embeddings = service.embed(texts)

# Combine chunks with embeddings
for chunk, embedding in zip(chunks, embeddings):
    chunk.embedding = embedding
```

### With Vector Database

```python
import faiss
import numpy as np

# Generate embeddings
service = EmbeddingService()
embeddings = service.embed(texts)

# Create FAISS index
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)

# Add embeddings to index
embedding_matrix = np.array(embeddings).astype('float32')
index.add(embedding_matrix)
```

## Command Line Usage

```bash
# Test service connection
python embedding_service.py test

# Perform health check
python embedding_service.py health

# Embed single text
python embedding_service.py embed "Your text here"

# Embed text from file
python embedding_service.py embed-file document.txt

# Show service statistics
python embedding_service.py stats
```

### Example Output

```bash
# Health check output
🌐 Service available: ✅
🤖 Model loaded: ✅
🧪 Embedding test: ✅
⏱️ Response time: 0.145s

# Embedding output
✅ Generated embedding with 768 dimensions
📊 First 5 values: [0.123, -0.456, 0.789, -0.234, 0.567]
```

## Troubleshooting

### Common Issues

**Service not available:**
- Check if Ollama is running: `ollama serve`
- Verify model is downloaded: `ollama pull nomic-embed-text`
- Check port accessibility: `curl http://localhost:11434/api/tags`

**Slow performance:**
- Reduce batch size
- Check system resources
- Verify network connection
- Consider model size

**Dimension mismatches:**
- Different models have different dimensions
- Verify model compatibility
- Check for model updates

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

service = EmbeddingService()
embeddings = service.embed(texts)
```

### Model Information

```python
service = EmbeddingService()
model_info = service.get_model_info()

if model_info:
    print(f"Model: {model_info.get('name')}")
    print(f"Size: {model_info.get('size')}")
    print(f"Modified: {model_info.get('modified_at')}")
```

## Convenience Functions

### Quick Embedding

```python
from modules.embedding_service import embed_text, embed_texts

# Single text
embedding = embed_text("Your text here")

# Multiple texts
embeddings = embed_texts(["Text 1", "Text 2", "Text 3"])
```

## Dependencies

- **requests**: HTTP client for Ollama API
- **time**: Performance timing
- **typing**: Type hints
- **warnings**: Warning suppression