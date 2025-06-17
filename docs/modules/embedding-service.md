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
    batch_size=20,
    show_progress=True
)
```

## API Reference

### EmbeddingService Class

#### `__init__(model="nomic-embed-text", base_url="http://localhost:11434", default_dimension=768, timeout=30, batch_size=10, show_progress=True)`

Initialize EmbeddingService with configuration.

**Parameters:**
- `model`: Ollama embedding model to use
- `base_url`: Ollama server URL  
- `default_dimension`: Default embedding dimension for fallback
- `timeout`: Request timeout in seconds
- `batch_size`: Number of texts to process in parallel
- `show_progress`: Whether to display progress bars during embedding

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

## Integration with ProcessorConfig

The EmbeddingService integrates seamlessly with the ModularPDFProcessor configuration system:

```python
from pdf_processor import ProcessorConfig, ModularPDFProcessor

# Configuration includes embedding service settings
config = ProcessorConfig(
    embedding_model="nomic-embed-text",      # Ollama embedding model
    ollama_base_url="http://localhost:11434", # Ollama server URL
    generate_embeddings=True,                 # Enable embedding generation
    max_workers=8,                           # Parallel processing workers
    verbose=True,                            # Detailed service logging
    force_reprocess=False                    # Skip already processed files
)

processor = ModularPDFProcessor(config)
# EmbeddingService automatically initialized with optimal settings
```

### ProcessorConfig Integration Points

- **embedding_model**: Specifies the Ollama model for embedding generation
- **ollama_base_url**: Ollama server endpoint configuration
- **generate_embeddings**: Controls whether embeddings are generated
- **max_workers**: Influences batch processing parallelization
- **verbose**: Controls detailed service logging and performance metrics
- **force_reprocess**: Determines whether to regenerate existing embeddings

## Enhanced Health Check System

### Comprehensive Service Monitoring

Advanced health checking with detailed diagnostics:

```python
def health_check(self) -> Dict[str, Any]:
    """Comprehensive health check with detailed diagnostics."""
    start_time = time.time()
    
    health_result = {
        "timestamp": time.time(),
        "service_available": False,
        "model_loaded": False,
        "embedding_test": False,
        "response_time": 0.0,
        "model_info": None,
        "api_version": None,
        "error": None,
        "recommendations": []
    }
    
    try:
        # Test 1: Basic connectivity
        response = requests.get(f"{self.base_url}/api/version", timeout=5)
        if response.status_code == 200:
            health_result["service_available"] = True
            health_result["api_version"] = response.json().get("version", "unknown")
        else:
            health_result["error"] = f"Service returned status {response.status_code}"
            health_result["recommendations"].append("Check Ollama service status")
            return health_result
        
        # Test 2: Model availability
        model_response = requests.get(f"{self.base_url}/api/tags", timeout=10)
        if model_response.status_code == 200:
            models = model_response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if self.model in model_names:
                health_result["model_loaded"] = True
                # Get detailed model info
                model_info = next((m for m in models if m.get("name") == self.model), {})
                health_result["model_info"] = {
                    "name": model_info.get("name"),
                    "size": model_info.get("size"),
                    "modified_at": model_info.get("modified_at"),
                    "digest": model_info.get("digest", "")[:12]  # Short digest
                }
            else:
                health_result["error"] = f"Model '{self.model}' not found"
                health_result["recommendations"].extend([
                    f"Run: ollama pull {self.model}",
                    "Check model name spelling"
                ])
                return health_result
        
        # Test 3: Embedding functionality
        test_embedding = self.embed_single("Health check test")
        if test_embedding and len(test_embedding) > 0:
            health_result["embedding_test"] = True
            health_result["embedding_dimension"] = len(test_embedding)
        else:
            health_result["error"] = "Embedding generation failed"
            health_result["recommendations"].append("Check model compatibility")
        
    except requests.exceptions.ConnectionError:
        health_result["error"] = "Cannot connect to Ollama service"
        health_result["recommendations"].extend([
            "Start Ollama service: ollama serve",
            "Check if port 11434 is accessible"
        ])
    except requests.exceptions.Timeout:
        health_result["error"] = "Service timeout"
        health_result["recommendations"].extend([
            "Check system resources",
            "Consider smaller batch sizes"
        ])
    except Exception as e:
        health_result["error"] = str(e)
        health_result["recommendations"].append("Check Ollama logs for details")
    
    health_result["response_time"] = time.time() - start_time
    
    # Add performance recommendations
    if health_result["response_time"] > 5.0:
        health_result["recommendations"].append("Consider faster hardware or smaller model")
    
    return health_result

def get_service_status(self) -> str:
    """Get simple service status string."""
    health = self.health_check()
    
    if health["embedding_test"]:
        return "✅ Healthy"
    elif health["model_loaded"]:
        return "⚠️ Model loaded but embedding failed"
    elif health["service_available"]:
        return "⚠️ Service available but model not loaded"
    else:
        return "❌ Service unavailable"
```

### Health Check Features

- **Multi-stage testing**: Comprehensive service validation
- **Model verification**: Confirms specific model availability
- **Performance measurement**: Response time monitoring
- **Error diagnosis**: Detailed error reporting with recommendations
- **Recovery suggestions**: Actionable troubleshooting steps

## Advanced Batch Processing

### Intelligent Batch Management

Optimized batch processing with adaptive sizing and error recovery:

```python
def embed(self, texts: List[str]) -> List[List[float]]:
    """Generate embeddings with intelligent batch processing."""
    if not texts:
        return []
    
    start_time = time.time()
    embeddings = []
    
    # Adaptive batch sizing based on text lengths
    adaptive_batch_size = self._calculate_optimal_batch_size(texts)
    
    # Process in batches
    for i in range(0, len(texts), adaptive_batch_size):
        batch_texts = texts[i:i + adaptive_batch_size]
        batch_embeddings = self._process_batch_with_retry(batch_texts)
        embeddings.extend(batch_embeddings)
        
        # Update statistics
        self.stats["total_requests"] += 1
        
        # Progress reporting for large batches
        if len(texts) > 50 and i % (adaptive_batch_size * 5) == 0:
            progress = (i + adaptive_batch_size) / len(texts) * 100
            print(f"📊 Embedding progress: {progress:.1f}% ({i + adaptive_batch_size}/{len(texts)})")
    
    # Update performance statistics
    processing_time = time.time() - start_time
    self.stats["total_processing_time"] += processing_time
    self.stats["total_texts_processed"] += len(texts)
    
    return embeddings

def _calculate_optimal_batch_size(self, texts: List[str]) -> int:
    """Calculate optimal batch size based on text characteristics."""
    if not texts:
        return self.batch_size
    
    # Analyze text lengths
    avg_length = sum(len(text) for text in texts) / len(texts)
    max_length = max(len(text) for text in texts)
    
    # Adjust batch size based on text complexity
    if avg_length > 2000:  # Long texts
        return max(1, self.batch_size // 2)
    elif avg_length < 500:  # Short texts
        return min(20, self.batch_size * 2)
    elif max_length > 5000:  # Mixed with very long texts
        return max(2, self.batch_size // 3)
    else:
        return self.batch_size

def _process_batch_with_retry(self, batch_texts: List[str], max_retries: int = 2) -> List[List[float]]:
    """Process batch with retry logic and error recovery."""
    for attempt in range(max_retries + 1):
        try:
            # Prepare batch request
            batch_data = {
                "model": self.model,
                "prompt": batch_texts if len(batch_texts) == 1 else batch_texts,
                "options": {
                    "temperature": 0,
                    "top_p": 0.9
                }
            }
            
            # Make API request
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=batch_data,
                timeout=self.timeout * (attempt + 1)  # Increase timeout on retries
            )
            
            if response.status_code == 200:
                result = response.json()
                embeddings = result.get("embeddings", [])
                
                if len(embeddings) == len(batch_texts):
                    self.stats["successful_requests"] += 1
                    return embeddings
                else:
                    print(f"⚠️ Embedding count mismatch: got {len(embeddings)}, expected {len(batch_texts)}")
            
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                print(f"⏱️ Batch timeout, retrying (attempt {attempt + 2}/{max_retries + 1})...")
                time.sleep(2 ** attempt)  # Exponential backoff
            continue
        except Exception as e:
            if attempt < max_retries:
                print(f"⚠️ Batch error: {e}, retrying...")
                time.sleep(1)
            continue
    
    # All retries failed, return fallback embeddings
    self.stats["failed_requests"] += 1
    self.stats["fallback_embeddings"] += len(batch_texts)
    
    print(f"❌ Batch processing failed after {max_retries + 1} attempts, using fallback")
    return [self._get_fallback_embedding() for _ in batch_texts]
```

### Batch Processing Features

- **Adaptive sizing**: Adjusts batch size based on text characteristics
- **Retry logic**: Handles transient failures with exponential backoff
- **Progress tracking**: Reports progress for large embedding jobs
- **Error recovery**: Graceful fallback for failed batches
- **Performance optimization**: Balances throughput and reliability

## Enhanced Performance Monitoring

### Detailed Statistics Tracking

```python
class EmbeddingService:
    def __init__(self, *args, **kwargs):
        # ... initialization ...
        self.stats = {
            # Request statistics
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retry_attempts": 0,
            
            # Processing statistics
            "total_texts_processed": 0,
            "total_processing_time": 0.0,
            "avg_texts_per_second": 0.0,
            
            # Quality statistics
            "fallback_embeddings": 0,
            "dimension_mismatches": 0,
            "timeout_count": 0,
            
            # Performance tracking
            "fastest_request": float('inf'),
            "slowest_request": 0.0,
            "avg_request_time": 0.0,
            
            # Health monitoring
            "last_health_check": 0.0,
            "consecutive_failures": 0,
            "service_uptime": 0.0
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        current_time = time.time()
        
        # Calculate derived metrics
        success_rate = 0.0
        if self.stats["total_requests"] > 0:
            success_rate = (self.stats["successful_requests"] / self.stats["total_requests"]) * 100
        
        texts_per_second = 0.0
        if self.stats["total_processing_time"] > 0:
            texts_per_second = self.stats["total_texts_processed"] / self.stats["total_processing_time"]
        
        fallback_rate = 0.0
        if self.stats["total_texts_processed"] > 0:
            fallback_rate = (self.stats["fallback_embeddings"] / self.stats["total_texts_processed"]) * 100
        
        return {
            "summary": {
                "success_rate": f"{success_rate:.1f}%",
                "processing_speed": f"{texts_per_second:.1f} texts/sec",
                "fallback_rate": f"{fallback_rate:.1f}%",
                "service_health": self.get_service_status()
            },
            "requests": {
                "total": self.stats["total_requests"],
                "successful": self.stats["successful_requests"],
                "failed": self.stats["failed_requests"],
                "retries": self.stats["retry_attempts"]
            },
            "processing": {
                "texts_processed": self.stats["total_texts_processed"],
                "total_time": f"{self.stats['total_processing_time']:.2f}s",
                "avg_request_time": f"{self.stats['avg_request_time']:.3f}s",
                "fastest_request": f"{self.stats['fastest_request']:.3f}s",
                "slowest_request": f"{self.stats['slowest_request']:.3f}s"
            },
            "quality": {
                "fallback_embeddings": self.stats["fallback_embeddings"],
                "dimension_mismatches": self.stats["dimension_mismatches"],
                "timeout_count": self.stats["timeout_count"]
            },
            "health": {
                "consecutive_failures": self.stats["consecutive_failures"],
                "last_health_check": self.stats["last_health_check"],
                "service_uptime": f"{(current_time - self.stats['service_uptime']):.0f}s" if self.stats['service_uptime'] > 0 else "Unknown"
            }
        }
```

### Performance Monitoring Features

- **Real-time metrics**: Continuous tracking of service performance
- **Quality indicators**: Monitors embedding quality and fallback usage
- **Health trending**: Tracks service health over time
- **Performance profiling**: Identifies bottlenecks and optimization opportunities
- **Alerting thresholds**: Can be configured to alert on performance degradation

## New Diagnostic Tools

### Advanced Troubleshooting

```python
def diagnose_performance_issues(self) -> Dict[str, Any]:
    """Diagnose common performance issues."""
    diagnosis = {
        "issues_found": [],
        "recommendations": [],
        "severity": "info"  # info, warning, critical
    }
    
    stats = self.get_statistics()
    
    # Check success rate
    if stats.get("success_rate", 100) < 90:
        diagnosis["issues_found"].append(f"Low success rate: {stats['success_rate']:.1f}%")
        diagnosis["recommendations"].append("Check Ollama service stability")
        diagnosis["severity"] = "warning"
    
    # Check processing speed
    if stats.get("texts_per_second", 0) < 1:
        diagnosis["issues_found"].append(f"Slow processing: {stats['texts_per_second']:.1f} texts/sec")
        diagnosis["recommendations"].extend([
            "Consider reducing batch size",
            "Check system resources (CPU/Memory)",
            "Verify model is quantized version"
        ])
        diagnosis["severity"] = "warning"
    
    # Check fallback rate
    fallback_rate = (self.stats["fallback_embeddings"] / max(1, self.stats["total_texts_processed"])) * 100
    if fallback_rate > 10:
        diagnosis["issues_found"].append(f"High fallback rate: {fallback_rate:.1f}%")
        diagnosis["recommendations"].extend([
            "Check network connectivity",
            "Verify model compatibility",
            "Consider increasing timeout values"
        ])
        diagnosis["severity"] = "critical" if fallback_rate > 25 else "warning"
    
    # Check timeout frequency
    if self.stats["timeout_count"] > self.stats["successful_requests"] * 0.1:
        diagnosis["issues_found"].append("Frequent timeouts detected")
        diagnosis["recommendations"].extend([
            "Increase timeout values",
            "Check system load",
            "Consider smaller batch sizes"
        ])
        diagnosis["severity"] = "warning"
    
    return diagnosis

def benchmark_performance(self, test_texts: List[str] = None) -> Dict[str, Any]:
    """Run performance benchmark."""
    if test_texts is None:
        test_texts = [
            "Short text for testing embedding performance.",
            "This is a medium-length text that should provide a good baseline for embedding generation performance testing and analysis.",
            "This is a longer text sample designed to test embedding generation performance with more substantial content. It includes multiple sentences and should provide insight into how the embedding service handles larger text chunks during batch processing operations."
        ]
    
    benchmark_results = {
        "test_config": {
            "num_texts": len(test_texts),
            "batch_size": self.batch_size,
            "model": self.model
        },
        "performance": {},
        "quality": {}
    }
    
    # Warm-up
    self.embed_single("Warm-up text")
    
    # Single embedding benchmark
    start_time = time.time()
    single_embedding = self.embed_single(test_texts[0])
    single_time = time.time() - start_time
    
    # Batch embedding benchmark
    start_time = time.time()
    batch_embeddings = self.embed(test_texts)
    batch_time = time.time() - start_time
    
    benchmark_results["performance"] = {
        "single_embedding_time": f"{single_time:.3f}s",
        "batch_embedding_time": f"{batch_time:.3f}s",
        "texts_per_second": f"{len(test_texts) / batch_time:.1f}",
        "avg_time_per_text": f"{batch_time / len(test_texts):.3f}s"
    }
    
    benchmark_results["quality"] = {
        "embedding_dimension": len(single_embedding) if single_embedding else 0,
        "all_embeddings_generated": len(batch_embeddings) == len(test_texts),
        "dimension_consistency": all(len(emb) == len(single_embedding) for emb in batch_embeddings) if batch_embeddings else False
    }
    
    return benchmark_results
```

## Dependencies

- **requests**: HTTP client for Ollama API (>= 2.25.0)
- **time**: Performance timing (Python standard library)
- **typing**: Type hints (Python standard library)
- **warnings**: Warning suppression (Python standard library)
- **json**: JSON handling (Python standard library)
- **statistics**: Statistical calculations (Python standard library)
- **concurrent.futures**: Parallel processing support (Python 3.2+)
- **pathlib**: Path handling (Python standard library)