# QRGenerator Module

## Overview

The `QRGenerator` module handles parallel QR code generation for text chunks with progress tracking, temporary file management, and optimized processing for video assembly with MemVid integration.

## Features

- **Parallel QR generation** using ProcessPoolExecutor
- **Progress tracking** with visual indicators
- **Frame validation** and quality assurance
- **MemVid integration** with monkey patching
- **Error recovery** and graceful degradation
- **Performance statistics** and monitoring

## Installation

```bash
pip install memvid qrcode pillow
```

## Basic Usage

### Generate QR Frames

```python
from modules.qr_generator import QRGenerator
from pathlib import Path

# Prepare chunk data
chunks = [
    {
        "text": "First chunk of text",
        "metadata": {"chunk_index": 0}
    },
    {
        "text": "Second chunk of text", 
        "metadata": {"chunk_index": 1}
    }
]

# Generate QR frames
generator = QRGenerator(n_workers=4)
temp_dir = Path("./temp")
frames_dir, stats = generator.generate_qr_frames(chunks, temp_dir)

print(f"Generated {stats['successful_frames']} frames")
print(f"Processing rate: {stats['frames_per_second']:.1f} frames/sec")
```

### Sequential Generation (Debugging)

```python
# Use sequential processing for debugging
generator = QRGenerator()
frames_dir, stats = generator.generate_qr_frames_sequential(chunks, temp_dir)
```

### MemVid Integration

```python
from memvid import MemvidEncoder

# Monkey patch encoder for parallel processing
encoder = MemvidEncoder()
generator = QRGenerator(n_workers=8)
generator.monkey_patch_memvid_encoder(encoder)

# Now encoder will use parallel QR generation
```

## API Reference

### QRGenerator Class

#### `__init__(n_workers=None, show_progress=True)`

Initialize QRGenerator with configuration.

**Parameters:**
- `n_workers`: Number of worker processes (None = auto-detect)
- `show_progress`: Whether to show progress bars

#### `generate_qr_frames(chunks, temp_dir) -> Tuple[Path, Dict]`

Generate QR frames for text chunks in parallel.

**Parameters:**
- `chunks`: List of chunk dictionaries with text and metadata
- `temp_dir`: Temporary directory for frame generation

**Returns:**
- Tuple of (frames_directory, generation_stats)

#### `generate_qr_frames_sequential(chunks, temp_dir) -> Tuple[Path, Dict]`

Generate QR frames sequentially (fallback for debugging).

**Parameters:**
- `chunks`: List of chunk dictionaries with text and metadata
- `temp_dir`: Temporary directory for frame generation

**Returns:**
- Tuple of (frames_directory, generation_stats)

#### `validate_frames(frames_dir) -> Dict[str, Any]`

Validate generated QR frames.

**Parameters:**
- `frames_dir`: Directory containing generated frames

**Returns:**
- Validation results dictionary

#### `monkey_patch_memvid_encoder(encoder)`

Monkey patch MemvidEncoder to use parallel QR generation.

**Parameters:**
- `encoder`: MemvidEncoder instance to patch

## Chunk Data Format

Each chunk should be a dictionary with the following structure:

```python
{
    "text": "The text content to encode in QR",
    "metadata": {
        "chunk_index": 0,
        "page_reference": "15",
        "title": "Document Title",
        # ... other metadata fields
    }
}
```

## Generation Statistics

The generator tracks comprehensive statistics:

```python
{
    "total_frames": 100,
    "successful_frames": 98,
    "failed_frames": 2,
    "processing_time": 15.7,
    "frames_per_second": 6.4
}
```

### Performance Metrics

- **total_frames**: Number of frames requested
- **successful_frames**: Frames generated successfully
- **failed_frames**: Frames that failed to generate
- **processing_time**: Total time in seconds
- **frames_per_second**: Processing throughput

## Parallel Processing

### Worker Process Management

```python
import multiprocessing

# Auto-detect optimal worker count
optimal_workers = multiprocessing.cpu_count()
generator = QRGenerator(n_workers=optimal_workers)

# Manual worker configuration
generator = QRGenerator(n_workers=8)  # Fixed 8 workers
```

### Process Pool Strategy

1. **Chunk preparation**: Format data for worker processes
2. **Process spawning**: Create worker pool with specified size
3. **Task distribution**: Distribute chunks across workers
4. **Result collection**: Gather results with progress tracking
5. **Error handling**: Handle individual worker failures
6. **Resource cleanup**: Clean up worker pool

### Memory Management

- Each worker process operates independently
- Minimal memory overhead per worker
- Automatic cleanup of worker processes
- Efficient data serialization for process communication

## Frame Validation

### Validation Checks

```python
generator = QRGenerator()
validation = generator.validate_frames(frames_dir)

if validation["valid"]:
    print(f"✅ {validation['frame_count']} frames validated")
else:
    print(f"❌ Validation failed: {validation['error']}")
```

### Validation Criteria

- **Existence check**: Frames directory exists
- **File count**: Expected number of frame files
- **Naming convention**: Correct frame_XXXXXX.png format
- **Sequence integrity**: No missing frame indices
- **File size**: Frames are not empty

## Error Handling

### Individual Frame Failures

```python
# Even if some frames fail, processing continues
chunks = [chunk1, chunk2, corrupted_chunk, chunk4]
frames_dir, stats = generator.generate_qr_frames(chunks, temp_dir)

print(f"Success rate: {stats['successful_frames']}/{stats['total_frames']}")
# Output: Success rate: 3/4
```

### Worker Process Errors

- **Process crashes**: Isolated failures don't affect other workers
- **Memory issues**: Individual processes can fail safely
- **Timeout handling**: Long-running workers are terminated
- **Resource exhaustion**: Graceful degradation with fewer workers

### Fallback Strategies

```python
try:
    # Try parallel processing first
    frames_dir, stats = generator.generate_qr_frames(chunks, temp_dir)
except Exception as e:
    print(f"Parallel processing failed: {e}")
    # Fall back to sequential processing
    frames_dir, stats = generator.generate_qr_frames_sequential(chunks, temp_dir)
```

## Integration Examples

### With TextChunker

```python
from modules.text_chunker import TextChunker
from modules.qr_generator import QRGenerator

# Create chunks
chunker = TextChunker()
chunks = chunker.create_enhanced_chunks(page_texts)

# Prepare chunk data for QR generation
chunk_data = []
for chunk in chunks:
    chunk_data.append({
        "text": chunk.text,
        "metadata": chunk.enhanced_metadata
    })

# Generate QR frames
generator = QRGenerator()
frames_dir, stats = generator.generate_qr_frames(chunk_data, temp_dir)
```

### With VideoAssembler

```python
from modules.qr_generator import QRGenerator
from modules.video_assembler import VideoAssembler

# Generate QR frames
generator = QRGenerator()
frames_dir, qr_stats = generator.generate_qr_frames(chunks, temp_dir)

# Assemble video from frames
assembler = VideoAssembler()
result = assembler.assemble_video(
    frames_dir, chunks, output_video, output_index
)
```

## Command Line Usage

```bash
# Test QR generation
python qr_generator.py test

# Generate from text file
python qr_generator.py generate document.txt

# Test parallel vs sequential
python qr_generator.py parallel document.txt
```

### Example Output

```bash
# Test output
⚡ Generating 3 QR frames using 4 workers...
✅ Generated 3/3 frames in 0.8s
📊 Rate: 3.8 frames/second

📊 Generation Statistics:
   total_frames: 3
   successful_frames: 3
   failed_frames: 0
   processing_time: 0.79
   frames_per_second: 3.81

✅ Validation: PASSED
```

## Performance Optimization

### Worker Count Selection

```python
import psutil

# Consider system resources
cpu_count = psutil.cpu_count()
memory_gb = psutil.virtual_memory().total / (1024**3)

# Conservative worker count for memory-constrained systems
if memory_gb < 8:
    workers = min(4, cpu_count)
else:
    workers = cpu_count

generator = QRGenerator(n_workers=workers)
```

### Batch Size Optimization

```python
# Optimal batch sizes based on chunk count
def optimal_workers(chunk_count):
    if chunk_count < 10:
        return 2  # Small batch
    elif chunk_count < 100:
        return 4  # Medium batch
    else:
        return 8  # Large batch

workers = optimal_workers(len(chunks))
generator = QRGenerator(n_workers=workers)
```

### Progress Monitoring

```python
from tqdm import tqdm

# Custom progress monitoring
generator = QRGenerator(show_progress=True)
frames_dir, stats = generator.generate_qr_frames(chunks, temp_dir)

# Progress bar will show:
# QR Generation: 100%|██████████| 150/150 [00:23<00:00, 6.4it/s]
```

## Troubleshooting

### Common Issues

**Workers failing to start:**
- Check multiprocessing support
- Verify MemVid installation
- Try sequential processing mode

**Memory issues:**
- Reduce worker count
- Process smaller batches
- Check system memory usage

**Import errors in workers:**
- Verify all dependencies installed
- Check Python path consistency
- Try sequential mode for debugging

### Debug Mode

```python
# Enable detailed logging for worker processes
import logging
logging.basicConfig(level=logging.DEBUG)

generator = QRGenerator(n_workers=1, show_progress=True)
frames_dir, stats = generator.generate_qr_frames(chunks, temp_dir)
```

### Frame Cleanup

```python
# Clean up generated frames
generator = QRGenerator()
success = generator.cleanup_frames(frames_dir)
if success:
    print("✅ Frames cleaned up successfully")
```

## Convenience Functions

### Quick QR Generation

```python
from modules.qr_generator import generate_qr_frames_from_texts

# Generate from text list
texts = ["Text 1", "Text 2", "Text 3"]
stats = generate_qr_frames_from_texts(texts, "./output", n_workers=4)
print(f"Generated {stats['successful_frames']} frames")
```

## Dependencies

- **multiprocessing**: Parallel processing
- **concurrent.futures**: ProcessPoolExecutor
- **memvid**: QR frame generation and video assembly
- **pathlib**: Path handling
- **tqdm**: Progress visualization
- **warnings**: Warning suppression