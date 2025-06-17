#!/usr/bin/env python3
"""
EmbeddingService Module

Handles text embedding generation using Ollama's nomic-embed-text model
with batch processing, error recovery, and dimension handling.
"""

import warnings
from typing import List, Optional, Dict, Any
import requests
import time
from tqdm import tqdm

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


class EmbeddingService:
    """Embedding service using Ollama's nomic-embed-text model."""
    
    def __init__(self,
                 model: str = "nomic-embed-text",
                 base_url: str = "http://localhost:11434",
                 default_dimension: int = 768,
                 timeout: int = 30,
                 batch_size: int = 10,
                 show_progress: bool = True):
        """Initialize EmbeddingService.
        
        Args:
            model: Ollama embedding model to use
            base_url: Ollama server URL
            default_dimension: Default embedding dimension for fallback
            timeout: Request timeout in seconds
            batch_size: Number of texts to process in parallel
        """
        self.model = model
        self.base_url = base_url
        self.default_dimension = default_dimension
        self.timeout = timeout
        self.batch_size = batch_size
        self.show_progress = show_progress
        
        # Track service statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_texts_processed": 0,
            "total_processing_time": 0.0,
            "fallback_embeddings": 0
        }
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (one per input text)
        """
        if not texts:
            return []
        
        embeddings = []
        start_time = time.time()
        
        # Process in batches for better performance with optional progress tracking
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        if self.show_progress:
            with tqdm(total=len(texts), desc="🧠 Generating embeddings", unit="texts",
                      bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i + self.batch_size]
                    batch_embeddings = self._process_batch(batch)
                    embeddings.extend(batch_embeddings)
                    pbar.update(len(batch))
        else:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self._process_batch(batch)
                embeddings.extend(batch_embeddings)
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats["total_texts_processed"] += len(texts)
        self.stats["total_processing_time"] += processing_time
        
        return embeddings
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector for the text
        """
        embeddings = self.embed([text])
        return embeddings[0] if embeddings else self._get_fallback_embedding()
    
    def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """Process a batch of texts for embedding.
        
        Args:
            texts: Batch of texts to process
            
        Returns:
            List of embeddings for the batch
        """
        embeddings = []
        
        for text in texts:
            try:
                embedding = self._get_embedding(text)
                embeddings.append(embedding)
                self.stats["successful_requests"] += 1
                
            except Exception as e:
                print(f"Error getting embedding for text: {e}")
                fallback = self._get_fallback_embedding()
                embeddings.append(fallback)
                self.stats["failed_requests"] += 1
                self.stats["fallback_embeddings"] += 1
            
            self.stats["total_requests"] += 1
        
        return embeddings
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text from Ollama.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        # Truncate very long texts to avoid API limits
        if len(text) > 8000:  # Conservative limit
            text = text[:8000] + "..."
        
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.model,
                "prompt": text
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        
        if "embedding" in result:
            embedding = result["embedding"]
            
            # Validate embedding dimension
            if len(embedding) != self.default_dimension:
                print(f"Warning: Unexpected embedding dimension: {len(embedding)} (expected {self.default_dimension})")
            
            return embedding
        else:
            raise Exception(f"No embedding in response for text: {text[:50]}...")
    
    def _get_fallback_embedding(self) -> List[float]:
        """Generate fallback embedding when API fails.
        
        Returns:
            Zero vector of default dimension
        """
        return [0.0] * self.default_dimension
    
    def test_connection(self) -> bool:
        """Test connection to Ollama embedding service.
        
        Returns:
            True if service is available, False otherwise
        """
        try:
            test_text = "test connection"
            embedding = self._get_embedding(test_text)
            return len(embedding) > 0
        except Exception as e:
            print(f"Embedding service connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the embedding model.
        
        Returns:
            Model information dictionary or None if unavailable
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            models = response.json().get("models", [])
            for model in models:
                if model.get("name", "").startswith(self.model):
                    return model
            
            return None
            
        except Exception as e:
            print(f"Error getting model info: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service usage statistics.
        
        Returns:
            Dictionary with service statistics
        """
        stats = self.stats.copy()
        
        # Calculate derived statistics
        if stats["total_requests"] > 0:
            stats["success_rate"] = (stats["successful_requests"] / stats["total_requests"]) * 100
        else:
            stats["success_rate"] = 0.0
        
        if stats["total_texts_processed"] > 0 and stats["total_processing_time"] > 0:
            stats["texts_per_second"] = stats["total_texts_processed"] / stats["total_processing_time"]
        else:
            stats["texts_per_second"] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset service statistics."""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_texts_processed": 0,
            "total_processing_time": 0.0,
            "fallback_embeddings": 0
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of the embedding service.
        
        Returns:
            Health check results
        """
        health = {
            "service_available": False,
            "model_loaded": False,
            "embedding_test": False,
            "response_time": None,
            "error": None
        }
        
        try:
            # Test basic connection
            start_time = time.time()
            
            # Check if service is responding
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                health["service_available"] = True
                
                # Check if our model is loaded
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                if any(name.startswith(self.model) for name in model_names):
                    health["model_loaded"] = True
                    
                    # Test embedding generation
                    try:
                        test_embedding = self.embed_single("test embedding")
                        if len(test_embedding) == self.default_dimension:
                            health["embedding_test"] = True
                    except Exception as e:
                        health["error"] = f"Embedding test failed: {e}"
                else:
                    health["error"] = f"Model {self.model} not found in loaded models"
            else:
                health["error"] = f"Service returned status {response.status_code}"
            
            health["response_time"] = time.time() - start_time
            
        except requests.exceptions.Timeout:
            health["error"] = "Service timeout"
        except requests.exceptions.ConnectionError:
            health["error"] = "Cannot connect to service"
        except Exception as e:
            health["error"] = f"Health check failed: {e}"
        
        return health


# Utility functions for standalone usage
def embed_texts(texts: List[str], model: str = "nomic-embed-text") -> List[List[float]]:
    """Convenience function to embed multiple texts.
    
    Args:
        texts: List of texts to embed
        model: Embedding model to use
        
    Returns:
        List of embedding vectors
    """
    service = EmbeddingService(model=model)
    return service.embed(texts)


def embed_text(text: str, model: str = "nomic-embed-text") -> List[float]:
    """Convenience function to embed single text.
    
    Args:
        text: Text to embed
        model: Embedding model to use
        
    Returns:
        Embedding vector
    """
    service = EmbeddingService(model=model)
    return service.embed_single(text)


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python embedding_service.py <command> [args]")
        print("Commands:")
        print("  test                    - Test service connection")
        print("  health                  - Perform health check")
        print("  embed <text>           - Embed single text")
        print("  embed-file <file>      - Embed text from file")
        print("  stats                  - Show service statistics")
        sys.exit(1)
    
    command = sys.argv[1]
    service = EmbeddingService()
    
    if command == "test":
        print("Testing embedding service connection...")
        if service.test_connection():
            print("✅ Service is working correctly")
        else:
            print("❌ Service test failed")
    
    elif command == "health":
        print("Performing health check...")
        health = service.health_check()
        
        print(f"🌐 Service available: {'✅' if health['service_available'] else '❌'}")
        print(f"🤖 Model loaded: {'✅' if health['model_loaded'] else '❌'}")
        print(f"🧪 Embedding test: {'✅' if health['embedding_test'] else '❌'}")
        
        if health['response_time']:
            print(f"⏱️ Response time: {health['response_time']:.3f}s")
        
        if health['error']:
            print(f"❌ Error: {health['error']}")
    
    elif command == "embed" and len(sys.argv) >= 3:
        text = sys.argv[2]
        print(f"Embedding text: {text[:50]}...")
        
        embedding = service.embed_single(text)
        print(f"✅ Generated embedding with {len(embedding)} dimensions")
        print(f"📊 First 5 values: {embedding[:5]}")
    
    elif command == "embed-file" and len(sys.argv) >= 3:
        file_path = sys.argv[2]
        print(f"Embedding text from file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            embedding = service.embed_single(text)
            print(f"✅ Generated embedding with {len(embedding)} dimensions")
            print(f"📊 First 5 values: {embedding[:5]}")
            
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    
    elif command == "stats":
        stats = service.get_statistics()
        print("📊 Service Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
    
    else:
        print("❌ Invalid command or missing arguments")
        sys.exit(1)