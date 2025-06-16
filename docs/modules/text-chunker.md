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

## Integration with ProcessorConfig

The TextChunker integrates seamlessly with the ModularPDFProcessor configuration system:

```python
from pdf_processor import ProcessorConfig, ModularPDFProcessor

# Configuration includes chunking parameters
config = ProcessorConfig(
    chunk_size=500,              # Target tokens per chunk
    overlap_percentage=0.15,     # 15% overlap between chunks
    cross_page_context=100,      # Tokens for cross-page chunks
    verbose=True,                # Detailed chunking statistics
    force_reprocess=False        # Skip already processed files
)

processor = ModularPDFProcessor(config)
# TextChunker automatically initialized with optimal settings
```

### ProcessorConfig Integration Points

- **chunk_size**: Target number of tokens per chunk (impacts RAG performance)
- **overlap_percentage**: Overlap ratio between adjacent chunks (0.0-1.0)
- **cross_page_context**: Size of cross-page chunks in tokens
- **verbose**: Controls detailed chunking statistics and analysis output
- **force_reprocess**: Determines whether to re-chunk already processed files

## EnhancedChunk Improvements

### Comprehensive Metadata Structure

The EnhancedChunk dataclass now includes extensive metadata for optimal RAG performance:

```python
@dataclass
class EnhancedChunk:
    """Enhanced chunk with comprehensive metadata."""
    text: str
    start_page: int
    end_page: int
    chunk_index: int
    token_count: int
    enhanced_metadata: Dict[str, Any]
    
    @property
    def length(self) -> int:
        """Character count of chunk text."""
        return len(self.text)
    
    @property
    def is_cross_page(self) -> bool:
        """Check if chunk spans multiple pages."""
        return self.enhanced_metadata.get('cross_page', False)
    
    @property
    def page_reference(self) -> str:
        """Get citation-ready page reference."""
        if self.start_page == self.end_page:
            return str(self.enhanced_metadata.get('display_page', self.start_page))
        else:
            start_display = self.enhanced_metadata.get('display_start_page', self.start_page)
            end_display = self.enhanced_metadata.get('display_end_page', self.end_page)
            return f"{start_display}-{end_display}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            'text': self.text,
            'start_page': self.start_page,
            'end_page': self.end_page,
            'chunk_index': self.chunk_index,
            'token_count': self.token_count,
            'length': self.length,
            'enhanced_metadata': self.enhanced_metadata
        }
```

### Enhanced Metadata Fields

Each chunk now includes comprehensive metadata for better search and citation:

```python
# Example enhanced_metadata structure
{
    "file_name": "document.pdf",
    "title": "Machine Learning Fundamentals",
    "authors": "John Smith, Jane Doe", 
    "publishers": "MIT Press",
    "year": "2024",
    "doi": "10.1234/example",
    "chunk_index": 42,
    "page_reference": "15-16",
    "source_page": 13,
    "display_page": 15,
    "display_start_page": 15,
    "display_end_page": 16,
    "token_count": 487,
    "character_count": 1943,
    "num_pages": 150,
    "cross_page": True,
    "sentence_count": 12,
    "paragraph_count": 3,
    "has_references": True,
    "has_equations": False,
    "text_density": 0.85,
    "chunk_quality_score": 0.92
}
```

## Advanced Cross-Page Context

### Intelligent Boundary Detection

Improved cross-page chunk creation with semantic awareness:

```python
def create_cross_page_chunks(self, page_texts: Dict[int, str], page_offset: int = 0) -> List[EnhancedChunk]:
    """Create intelligent cross-page chunks."""
    cross_page_chunks = []
    pages = sorted(page_texts.keys())
    
    for i in range(len(pages) - 1):
        current_page = pages[i]
        next_page = pages[i + 1]
        
        current_text = page_texts[current_page]
        next_text = page_texts[next_page]
        
        # Extract context from end of current page
        current_tokens = self.tokenizer.encode(current_text)
        context_start = max(0, len(current_tokens) - self.cross_page_context // 2)
        current_context = self.tokenizer.decode(current_tokens[context_start:])
        
        # Extract context from beginning of next page
        next_tokens = self.tokenizer.encode(next_text)
        context_end = min(len(next_tokens), self.cross_page_context // 2)
        next_context = self.tokenizer.decode(next_tokens[:context_end])
        
        # Combine contexts with page boundary marker
        combined_text = f"{current_context}\n\n[PAGE BREAK]\n\n{next_context}"
        
        # Create enhanced cross-page chunk
        chunk = EnhancedChunk(
            text=combined_text,
            start_page=current_page,
            end_page=next_page,
            chunk_index=-1,  # Will be set later
            token_count=len(self.tokenizer.encode(combined_text)),
            enhanced_metadata={
                "cross_page": True,
                "source_page": current_page,
                "display_start_page": current_page + page_offset,
                "display_end_page": next_page + page_offset,
                "page_reference": f"{current_page + page_offset}-{next_page + page_offset}",
                "boundary_type": "page_break",
                "context_preservation": True
            }
        )
        
        cross_page_chunks.append(chunk)
    
    return cross_page_chunks
```

### Cross-Page Features

- **Semantic boundary detection**: Identifies natural break points
- **Context preservation**: Maintains narrative flow across pages
- **Page break annotation**: Clearly marks page boundaries in text
- **Balanced context**: Equal content from both pages
- **Quality scoring**: Evaluates cross-page chunk usefulness

## Improved Semantic Boundaries

### Sentence-Aware Chunking

Advanced boundary optimization that respects sentence structure:

```python
def optimize_chunk_boundary(self, text: str, target_end: int) -> int:
    """Find optimal chunk boundary near target position."""
    # Try to end at sentence boundary
    sentences = self._split_into_sentences(text[:target_end + 100])
    
    best_end = target_end
    min_distance = float('inf')
    
    current_pos = 0
    for sentence in sentences:
        sentence_end = current_pos + len(sentence)
        distance = abs(sentence_end - target_end)
        
        # Prefer sentence boundaries within reasonable range
        if distance < min_distance and distance <= self.chunk_size * 0.1:
            min_distance = distance
            best_end = sentence_end
        
        current_pos = sentence_end
        
        # Don't go too far past target
        if sentence_end > target_end + self.chunk_size * 0.2:
            break
    
    return best_end

def _split_into_sentences(self, text: str) -> List[str]:
    """Split text into sentences with academic paper awareness."""
    # Handle academic paper sentence patterns
    text = re.sub(r'\b(Dr|Prof|Mr|Mrs|Ms|PhD|et al)\.', r'\1 ', text)  # Protect titles
    text = re.sub(r'\b(Fig|Table|Eq|Ref)\.\s*(\d+)', r'\1 \2', text)  # Protect references
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Restore protected periods
    sentences = [s.replace(' ', '.') for s in sentences]
    
    return [s.strip() for s in sentences if s.strip()]
```

### Boundary Optimization Features

- **Sentence preservation**: Avoids cutting sentences mid-way
- **Academic formatting**: Handles references, figures, and equations
- **Paragraph awareness**: Prefers paragraph boundaries when possible
- **Quality scoring**: Evaluates boundary quality for optimization
- **Fallback mechanisms**: Graceful degradation for difficult text

## Enhanced Statistics and Analysis

### Comprehensive Chunking Metrics

```python
def get_enhanced_chunk_stats(self, chunks: List[EnhancedChunk]) -> Dict[str, Any]:
    """Get comprehensive chunking statistics."""
    if not chunks:
        return {}
    
    token_counts = [chunk.token_count for chunk in chunks]
    char_counts = [chunk.length for chunk in chunks]
    cross_page_chunks = [c for c in chunks if c.is_cross_page]
    
    # Quality metrics
    quality_scores = [chunk.enhanced_metadata.get('chunk_quality_score', 0.0) for chunk in chunks]
    
    # Boundary analysis
    sentence_boundaries = sum(1 for c in chunks if c.enhanced_metadata.get('ends_on_sentence', False))
    paragraph_boundaries = sum(1 for c in chunks if c.enhanced_metadata.get('ends_on_paragraph', False))
    
    return {
        # Basic statistics
        "total_chunks": len(chunks),
        "cross_page_chunks": len(cross_page_chunks),
        "regular_chunks": len(chunks) - len(cross_page_chunks),
        
        # Token statistics
        "token_stats": {
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "median_tokens": sorted(token_counts)[len(token_counts) // 2],
            "std_tokens": self._calculate_std(token_counts)
        },
        
        # Character statistics
        "character_stats": {
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
            "avg_chars": sum(char_counts) / len(char_counts),
            "total_chars": sum(char_counts)
        },
        
        # Quality metrics
        "quality_metrics": {
            "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            "sentence_boundary_rate": sentence_boundaries / len(chunks),
            "paragraph_boundary_rate": paragraph_boundaries / len(chunks),
            "optimal_size_rate": sum(1 for t in token_counts if abs(t - self.chunk_size) <= self.chunk_size * 0.1) / len(chunks)
        },
        
        # Distribution analysis
        "size_distribution": self._analyze_size_distribution(token_counts),
        "page_coverage": self._analyze_page_coverage(chunks)
    }

def _analyze_size_distribution(self, token_counts: List[int]) -> Dict[str, int]:
    """Analyze chunk size distribution."""
    size_ranges = {
        "very_small": sum(1 for t in token_counts if t < self.chunk_size * 0.5),
        "small": sum(1 for t in token_counts if self.chunk_size * 0.5 <= t < self.chunk_size * 0.8),
        "optimal": sum(1 for t in token_counts if self.chunk_size * 0.8 <= t <= self.chunk_size * 1.2),
        "large": sum(1 for t in token_counts if self.chunk_size * 1.2 < t <= self.chunk_size * 1.5),
        "very_large": sum(1 for t in token_counts if t > self.chunk_size * 1.5)
    }
    return size_ranges
```

### Advanced Analysis Features

- **Quality scoring**: Evaluates chunk semantic coherence
- **Boundary analysis**: Tracks sentence/paragraph boundary adherence
- **Size distribution**: Analyzes chunk size patterns
- **Page coverage**: Maps chunk distribution across document
- **Overlap validation**: Ensures proper overlap between chunks

## Performance Optimizations

### Memory-Efficient Processing

```python
def create_enhanced_chunks(self, page_texts: Dict[int, str], page_offset: int = 0) -> List[EnhancedChunk]:
    """Memory-efficient chunk creation."""
    all_chunks = []
    chunk_index = 0
    
    # Process pages in order for memory efficiency
    pages = sorted(page_texts.keys())
    
    for page_num in pages:
        text = page_texts[page_num]
        
        # Create chunks for this page
        page_chunks = self._create_page_chunks(text, page_num, page_offset, chunk_index)
        all_chunks.extend(page_chunks)
        chunk_index += len(page_chunks)
        
        # Memory cleanup
        del text
    
    # Add cross-page chunks
    cross_page_chunks = self.create_cross_page_chunks(page_texts, page_offset)
    for chunk in cross_page_chunks:
        chunk.chunk_index = chunk_index
        chunk_index += 1
    all_chunks.extend(cross_page_chunks)
    
    # Final metadata enhancement
    self._enhance_chunk_metadata(all_chunks)
    
    return all_chunks

def _enhance_chunk_metadata(self, chunks: List[EnhancedChunk]):
    """Add computed metadata to all chunks."""
    for chunk in chunks:
        # Add quality metrics
        chunk.enhanced_metadata.update({
            "character_count": chunk.length,
            "sentence_count": len(self._split_into_sentences(chunk.text)),
            "paragraph_count": len([p for p in chunk.text.split('\n\n') if p.strip()]),
            "has_references": bool(re.search(r'\[[0-9,\-\s]+\]', chunk.text)),
            "has_equations": bool(re.search(r'\$[^$]+\$|\\[a-zA-Z]+', chunk.text)),
            "text_density": len(chunk.text.replace(' ', '')) / len(chunk.text) if chunk.text else 0,
            "chunk_quality_score": self._calculate_chunk_quality(chunk)
        })

def _calculate_chunk_quality(self, chunk: EnhancedChunk) -> float:
    """Calculate chunk quality score (0.0-1.0)."""
    score = 0.0
    
    # Size optimality (0.3 weight)
    size_ratio = chunk.token_count / self.chunk_size
    if 0.8 <= size_ratio <= 1.2:
        score += 0.3
    elif 0.6 <= size_ratio <= 1.4:
        score += 0.2
    elif 0.5 <= size_ratio <= 1.5:
        score += 0.1
    
    # Sentence completeness (0.2 weight)
    if chunk.text.strip().endswith(('.', '!', '?', ':', ';')):
        score += 0.2
    
    # Content density (0.2 weight)
    text_density = chunk.enhanced_metadata.get('text_density', 0)
    if text_density > 0.7:
        score += 0.2
    elif text_density > 0.5:
        score += 0.1
    
    # Semantic coherence (0.3 weight)
    # Simple heuristic: paragraph count vs length ratio
    para_count = chunk.enhanced_metadata.get('paragraph_count', 1)
    if para_count <= 3:  # Coherent chunk
        score += 0.3
    elif para_count <= 5:
        score += 0.2
    else:
        score += 0.1
    
    return min(score, 1.0)
```

## New Configuration Options

### Advanced Chunking Parameters

```python
@dataclass
class AdvancedChunkingConfig:
    """Advanced chunking configuration options."""
    
    # Core parameters (inherited from ProcessorConfig)
    chunk_size: int = 500
    overlap_percentage: float = 0.15
    cross_page_context: int = 100
    
    # Advanced boundary optimization
    prefer_sentence_boundaries: bool = True
    prefer_paragraph_boundaries: bool = True
    max_boundary_deviation: float = 0.2  # As fraction of chunk_size
    
    # Quality control
    min_chunk_size: int = 50           # Minimum viable chunk size
    max_chunk_size: int = 1000         # Maximum allowed chunk size
    quality_threshold: float = 0.5      # Minimum quality score
    
    # Content filtering
    skip_short_paragraphs: bool = True
    skip_reference_lists: bool = False
    skip_table_of_contents: bool = True
    
    # Cross-page settings
    cross_page_enabled: bool = True
    cross_page_balance: float = 0.5     # Balance between pages (0.0-1.0)
    cross_page_marker: str = "[PAGE BREAK]"
```

## Dependencies

- **tiktoken**: Tokenization (GPT-4 compatible, >= 0.5.0)
- **dataclasses**: Enhanced chunk structure (Python 3.7+)
- **typing**: Type hints (Python standard library)
- **re**: Text processing regex operations (Python standard library)
- **pathlib**: Path handling (Python standard library)
- **json**: Serialization support (Python standard library)
- **statistics**: Statistical calculations (Python standard library)