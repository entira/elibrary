# TextChunker Module

## Overview

The `TextChunker` module handles intelligent text chunking with token-based sliding windows, cross-page context preservation, and semantic boundary optimization for optimal RAG performance.

## Features

- **Token-based chunking** with configurable size and overlap
- **Cross-page context preservation** for better semantic coherence
- **Semantic boundary optimization** (sentence-aware splitting)
- **Enhanced metadata** for each chunk with page references
- **Sliding window overlap** to prevent context loss
- **Statistics tracking** for performance analysis

## Installation

```bash
pip install tiktoken
```

## Basic Usage

### Simple Chunking

```python
from modules.text_chunker import TextChunker

# Page texts from TextExtractor
page_texts = {
    1: "First page content...",
    2: "Second page content...",
    3: "Third page content..."
}

chunker = TextChunker(chunk_size=500, overlap_percentage=0.15)
chunks = chunker.create_enhanced_chunks(page_texts, page_offset=0)

print(f"Created {len(chunks)} chunks")
for chunk in chunks:
    print(f"Chunk {chunk.chunk_index}: {chunk.token_count} tokens")
```

### Advanced Configuration

```python
from modules.text_chunker import TextChunker

# Custom chunking configuration
chunker = TextChunker(
    chunk_size=750,           # Larger chunks
    overlap_percentage=0.20,  # More overlap
    cross_page_context=150    # More cross-page context
)

chunks = chunker.create_enhanced_chunks(page_texts, page_offset=2)
```

### Convenience Function

```python
from modules.text_chunker import chunk_text

# Quick chunking with defaults
chunks = chunk_text(page_texts, chunk_size=500)
```

## API Reference

### EnhancedChunk Class

A dataclass representing a text chunk with comprehensive metadata.

**Attributes:**
- `text`: The chunk text content
- `start_page`: Starting page number
- `end_page`: Ending page number
- `chunk_index`: Sequential chunk index
- `token_count`: Number of tokens in chunk
- `enhanced_metadata`: Dictionary with detailed metadata
- `length`: Character count (auto-calculated)

### TextChunker Class

#### `__init__(chunk_size=500, overlap_percentage=0.15, cross_page_context=100)`

Initialize the TextChunker with configuration.

**Parameters:**
- `chunk_size`: Target tokens per chunk (default: 500)
- `overlap_percentage`: Overlap between chunks 0.0-1.0 (default: 0.15)
- `cross_page_context`: Tokens for cross-page chunks (default: 100)

#### `create_enhanced_chunks(page_texts, page_offset=0) -> List[EnhancedChunk]`

Create chunks with enhanced page metadata.

**Parameters:**
- `page_texts`: Dictionary mapping page numbers to text
- `page_offset`: Page numbering offset for citation accuracy

**Returns:**
- List of EnhancedChunk objects with comprehensive metadata

#### `get_chunk_stats(chunks) -> Dict[str, any]`

Get statistics about the chunking process.

**Parameters:**
- `chunks`: List of chunks to analyze

**Returns:**
- Dictionary with chunking statistics

## Enhanced Metadata

Each chunk includes comprehensive metadata:

```python
{
    "file_name": "document.pdf",
    "title": "Document Title", 
    "authors": "Author Names",
    "publishers": "Publisher Name",
    "year": "2024",
    "doi": "10.1234/example",
    "chunk_index": 42,
    "page_reference": "15",
    "source_page": 13,
    "display_page": 15,
    "token_count": 487,
    "num_pages": 150,
    "cross_page": False
}
```

## Chunking Strategy

### Token-Based Sliding Window

1. **Tokenization**: Uses GPT-4 tokenizer for consistency
2. **Window sizing**: Creates chunks of target token size
3. **Overlap calculation**: Maintains configurable overlap
4. **Boundary optimization**: Tries to end at sentence boundaries

### Cross-Page Chunks

Special chunks that span page boundaries:

- **Purpose**: Preserve context across page breaks
- **Content**: End of current page + beginning of next page
- **Identification**: Marked with `cross_page: True` in metadata
- **Size**: Configurable with `cross_page_context` parameter

### Semantic Boundary Optimization

- **Sentence-aware**: Tries to end chunks at sentence boundaries
- **Word preservation**: Avoids cutting words in half
- **Paragraph respect**: Maintains paragraph structure where possible
- **Length threshold**: Ensures chunks meet minimum length requirements

## Statistics and Analysis

### Chunk Statistics

```python
chunker = TextChunker()
chunks = chunker.create_enhanced_chunks(page_texts)
stats = chunker.get_chunk_stats(chunks)

print(f"Total chunks: {stats['total_chunks']}")
print(f"Cross-page chunks: {stats['cross_page_chunks']}")
print(f"Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
print(f"Token range: {stats['min_tokens']}-{stats['max_tokens']}")
```

### Performance Metrics

- **Total chunks**: Number of chunks created
- **Cross-page chunks**: Number of boundary-spanning chunks
- **Token distribution**: Min, max, and average tokens per chunk
- **Character distribution**: Text length statistics
- **Processing efficiency**: Tokens per second

## Configuration Guidelines

### Chunk Size Selection

**Small chunks (200-300 tokens):**
- Better for precise retrieval
- More chunks = more overhead
- Good for FAQ-style content

**Medium chunks (400-600 tokens):**
- Balanced approach (recommended)
- Good context preservation
- Suitable for most documents

**Large chunks (700+ tokens):**
- Better context preservation
- Fewer chunks = less overhead
- Good for narrative content

### Overlap Configuration

**Low overlap (5-10%):**
- Minimal redundancy
- Risk of context loss
- Faster processing

**Medium overlap (15-20%):**
- Good balance (recommended)
- Context preservation
- Moderate redundancy

**High overlap (25%+):**
- Maximum context preservation
- Higher redundancy
- More storage required

## Integration Examples

### With TextExtractor

```python
from modules.text_extractor import TextExtractor
from modules.text_chunker import TextChunker

# Extract text and create chunks
extractor = TextExtractor()
page_texts, num_pages, offset = extractor.extract_text_with_pages(pdf_path)

chunker = TextChunker()
chunks = chunker.create_enhanced_chunks(page_texts, offset)
```

### With Metadata Enhancement

```python
from modules.metadata_extractor import MetadataExtractor

# Extract metadata
metadata_extractor = MetadataExtractor()
metadata = metadata_extractor.extract_metadata(first_page_text, filename)

# Enhance chunks with metadata
for chunk in chunks:
    chunk.enhanced_metadata.update({
        "file_name": filename,
        "title": metadata["title"],
        "authors": metadata["authors"],
        # ... other metadata fields
    })
```

## Command Line Usage

```bash
# Chunk text from file
python text_chunker.py document.txt

# Chunk from stdin
echo "Your text here" | python text_chunker.py -

# Example output:
# 📊 Chunking Statistics:
#    Total chunks: 15
#    Cross-page chunks: 2
#    Average tokens per chunk: 487.3
#    Token range: 245-612
```

## Advanced Features

### Custom Tokenizer

```python
import tiktoken

# Use custom tokenizer
custom_tokenizer = tiktoken.get_encoding("cl100k_base")
chunker = TextChunker()
chunker.tokenizer = custom_tokenizer
```

### Chunk Validation

```python
# Validate chunk quality
for chunk in chunks:
    if chunk.token_count < 50:
        print(f"Warning: Very short chunk {chunk.chunk_index}")
    
    if chunk.token_count > 800:
        print(f"Warning: Very long chunk {chunk.chunk_index}")
```

### Cross-Page Analysis

```python
# Analyze cross-page chunks
cross_page_chunks = [c for c in chunks if c.enhanced_metadata.get('cross_page')]
print(f"Cross-page chunks: {len(cross_page_chunks)}")

for chunk in cross_page_chunks:
    print(f"Pages {chunk.start_page}-{chunk.end_page}: {chunk.token_count} tokens")
```

## Troubleshooting

### Common Issues

**Too many small chunks:**
- Increase minimum chunk size
- Adjust boundary optimization
- Check for fragmented text

**Token count mismatches:**
- Verify tokenizer consistency
- Check for encoding issues
- Validate chunk boundaries

**Poor cross-page chunks:**
- Adjust cross_page_context size
- Check page boundary detection
- Verify page text quality

### Debug Mode

```python
# Enable detailed chunk inspection
chunker = TextChunker()
chunks = chunker.create_enhanced_chunks(page_texts)

for i, chunk in enumerate(chunks[:5]):  # First 5 chunks
    print(f"Chunk {i}:")
    print(f"  Tokens: {chunk.token_count}")
    print(f"  Pages: {chunk.start_page}-{chunk.end_page}")
    print(f"  Preview: {chunk.text[:100]}...")
    print()
```

## Dependencies

- **tiktoken**: Tokenization (GPT-4 compatible)
- **dataclasses**: Enhanced chunk structure
- **typing**: Type hints
- **re**: Text processing regex operations