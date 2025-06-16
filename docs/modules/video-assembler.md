# VideoAssembler Module

## Overview

The `VideoAssembler` module handles video assembly from QR frames, index creation, and metadata enhancement with MemVid integration and comprehensive error handling for the final stage of the PDF-to-video pipeline.

## Features

- **MemVid integration** for video assembly
- **Enhanced index creation** with comprehensive metadata
- **Video validation** and quality assurance
- **Configurable output settings** (FPS, quality, compression)
- **Statistics tracking** and performance monitoring
- **Error recovery** and graceful degradation

## Installation

```bash
pip install memvid
```

## Basic Usage

### Assemble Video from Frames

```python
from modules.video_assembler import VideoAssembler
from pathlib import Path

# Prepare chunk data
chunks = [
    {
        "text": "First chunk content",
        "metadata": {
            "chunk_index": 0,
            "page_reference": "1",
            "title": "Document Title"
        }
    },
    # ... more chunks
]

# Assemble video
assembler = VideoAssembler(fps=30, quality="medium")
result = assembler.assemble_video(
    frames_dir=Path("./frames"),
    chunks=chunks,
    output_video=Path("./output/library.mp4"),
    output_index=Path("./output/library_index.json"),
    page_offsets={"document.pdf": 2}
)

if result["success"]:
    print(f"✅ Video created: {result['video_path']}")
    print(f"📋 Index created: {result['index_path']}")
else:
    print(f"❌ Assembly failed: {result['error']}")
```

### Custom Configuration

```python
assembler = VideoAssembler(
    fps=60,              # High frame rate
    quality="high",      # High quality output
    compression=True     # Enable compression
)
```

## API Reference

### VideoAssembler Class

#### `__init__(fps=30, quality="medium", compression=True)`

Initialize VideoAssembler with configuration.

**Parameters:**
- `fps`: Frames per second for video output
- `quality`: Video quality setting ("low", "medium", "high")
- `compression`: Whether to compress video output

#### `assemble_video(frames_dir, chunks, output_video, output_index, page_offsets=None) -> Dict`

Assemble video from QR frames and create enhanced index.

**Parameters:**
- `frames_dir`: Directory containing QR frame images
- `chunks`: List of chunk data with text and metadata
- `output_video`: Path for output video file
- `output_index`: Path for output index JSON file
- `page_offsets`: Optional page offset information

**Returns:**
- Assembly results and statistics dictionary

#### `create_enhanced_index(chunks, output_path, page_offsets=None) -> Dict`

Create enhanced index with comprehensive metadata.

**Parameters:**
- `chunks`: List of chunk data
- `output_path`: Path for output index file
- `page_offsets`: Optional page offset information

**Returns:**
- Index creation results dictionary

#### `validate_video_output(video_path) -> Dict[str, Any]`

Validate video output file.

**Parameters:**
- `video_path`: Path to video file

**Returns:**
- Validation results dictionary

#### `validate_index_output(index_path) -> Dict[str, Any]`

Validate index output file.

**Parameters:**
- `index_path`: Path to index file

**Returns:**
- Validation results dictionary

## Enhanced Index Structure

The module creates comprehensive JSON indexes with the following structure:

```json
{
  "metadata": [
    {
      "text": "Chunk text content",
      "length": 487,
      "enhanced_metadata": {
        "file_name": "document.pdf",
        "title": "Document Title",
        "authors": "Author Names",
        "publishers": "Publisher",
        "year": "2024",
        "doi": "10.1234/example",
        "chunk_index": 0,
        "page_reference": "15",
        "token_count": 123,
        "cross_page": false
      }
    }
  ],
  "enhanced_stats": {
    "total_files": 1,
    "total_chunks": 150,
    "cross_page_chunks": 12,
    "total_pages": 200,
    "total_text_length": 125432,
    "avg_chunk_length": 836.2,
    "total_tokens": 15678,
    "avg_tokens_per_chunk": 104.5,
    "files_processed": ["document.pdf"],
    "page_offsets": {"document.pdf": 2}
  },
  "version": "2.0",
  "created_by": "VideoAssembler",
  "assembly_config": {
    "fps": 30,
    "quality": "medium",
    "compression": true
  }
}
```

## Video Assembly Process

### Assembly Flow

1. **Frame validation**: Verify QR frames exist and are complete
2. **MemVid initialization**: Create encoder instance
3. **Chunk loading**: Add text chunks to encoder
4. **Video generation**: Build MP4 video with specified settings
5. **Index enhancement**: Create comprehensive metadata index
6. **Validation**: Verify output files are valid
7. **Statistics**: Calculate performance metrics

### Quality Settings

**Low Quality:**
- Smaller file size
- Faster processing
- Good for testing

**Medium Quality (Recommended):**
- Balanced file size and quality
- Optimal for most use cases
- Good compression ratio

**High Quality:**
- Larger file size
- Better visual quality
- Slower processing

## Statistics and Monitoring

### Assembly Statistics

```python
assembler = VideoAssembler()
result = assembler.assemble_video(frames_dir, chunks, video_path, index_path)

stats = result["stats"]
print(f"Video created: {stats['video_created']}")
print(f"Index created: {stats['index_created']}")
print(f"Total chunks: {stats['total_chunks']}")
print(f"Video duration: {stats['video_duration']:.1f}s")
print(f"File size: {stats['file_size_mb']:.1f} MB")
```

### Enhanced Statistics

The index includes comprehensive statistics:

- **total_files**: Number of source PDF files
- **total_chunks**: Total text chunks processed
- **cross_page_chunks**: Chunks spanning page boundaries
- **total_pages**: Total pages across all files
- **text_statistics**: Length and token metrics
- **processing_metadata**: File names and page offsets

## Validation and Quality Assurance

### Video Validation

```python
assembler = VideoAssembler()
validation = assembler.validate_video_output(video_path)

if validation["valid"]:
    print(f"✅ Video is valid ({validation['file_size_mb']:.1f} MB)")
    print(f"Appears to be video: {validation['appears_to_be_video']}")
else:
    print(f"❌ Video validation failed: {validation['error']}")
```

### Index Validation

```python
validation = assembler.validate_index_output(index_path)

if validation["valid"]:
    print(f"✅ Index is valid ({validation['metadata_entries']} entries)")
    print(f"Version: {validation['version']}")
else:
    print(f"❌ Index validation failed: {validation['error']}")
```

## Error Handling

### Assembly Errors

```python
try:
    result = assembler.assemble_video(frames_dir, chunks, video_path, index_path)
    if not result["success"]:
        print(f"Assembly failed: {result['error']}")
        # Handle specific error types
        if "MemVid" in result["error"]:
            print("MemVid integration issue")
        elif "frames" in result["error"]:
            print("Frame processing issue")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Graceful Degradation

- **Missing frames**: Skip missing frames, continue processing
- **MemVid errors**: Detailed error reporting with context
- **File system errors**: Proper error messages for disk issues
- **Memory issues**: Statistics tracking for resource monitoring

## Integration Examples

### Complete Pipeline

```python
from modules.text_extractor import TextExtractor
from modules.text_chunker import TextChunker
from modules.metadata_extractor import MetadataExtractor
from modules.qr_generator import QRGenerator
from modules.video_assembler import VideoAssembler

# Extract text
extractor = TextExtractor()
page_texts, num_pages, offset = extractor.extract_text_with_pages(pdf_path)

# Extract metadata
metadata_extractor = MetadataExtractor()
first_page = extractor.extract_first_page_text(pdf_path)
metadata = metadata_extractor.extract_metadata(first_page, pdf_path.name)

# Create chunks
chunker = TextChunker()
chunks = chunker.create_enhanced_chunks(page_texts, offset)

# Enhance chunks with metadata
for chunk in chunks:
    chunk.enhanced_metadata.update(metadata)

# Generate QR frames
generator = QRGenerator()
chunk_data = [{"text": c.text, "metadata": c.enhanced_metadata} for c in chunks]
frames_dir, qr_stats = generator.generate_qr_frames(chunk_data, temp_dir)

# Assemble video
assembler = VideoAssembler()
result = assembler.assemble_video(frames_dir, chunk_data, video_path, index_path)
```

### MemVid Direct Integration

```python
from memvid import MemvidEncoder

# Create library index using existing encoder
encoder = MemvidEncoder()
# ... add chunks to encoder ...

assembler = VideoAssembler()
success = assembler.create_library_index(
    encoder, video_path, index_path, page_offsets
)
```

## Command Line Usage

```bash
# Validate video file
python video_assembler.py validate-video library.mp4

# Validate index file
python video_assembler.py validate-index library_index.json

# Test assembly with sample data
python video_assembler.py test-assembly
```

### Example Output

```bash
# Validation output
✅ Video file is valid
   File size: 25.7 MB
   Appears to be video: True

✅ Index file is valid
   Metadata entries: 150
   Has enhanced stats: True
   Version: 2.0

# Test assembly output
✅ Index creation test passed
   Entries: 2
   Validation: PASSED
```

## Performance Considerations

### Memory Usage

- **Streaming processing**: Minimizes memory footprint
- **Efficient encoding**: MemVid handles large video assembly
- **Index optimization**: JSON structure optimized for size

### Processing Speed

- **Parallel frame processing**: Handled by QRGenerator
- **Optimized encoding**: MemVid native performance
- **Minimal overhead**: Efficient metadata processing

### File Size Optimization

```python
# Optimize for file size
assembler = VideoAssembler(
    fps=24,              # Lower frame rate
    quality="medium",    # Balanced quality
    compression=True     # Enable compression
)
```

## Troubleshooting

### Common Issues

**MemVid import errors:**
- Verify MemVid installation: `pip install memvid`
- Check Python environment compatibility
- Try importing MemVid directly for testing

**Video assembly failures:**
- Check frame directory exists and has frames
- Verify frame naming convention (frame_XXXXXX.png)
- Ensure sufficient disk space

**Index creation issues:**
- Verify chunk data structure
- Check file permissions for output directory
- Validate JSON serialization

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

assembler = VideoAssembler()
result = assembler.assemble_video(frames_dir, chunks, video_path, index_path)
```

### Resource Monitoring

```python
import psutil

# Monitor memory usage during assembly
process = psutil.Process()
memory_before = process.memory_info().rss / 1024 / 1024  # MB

result = assembler.assemble_video(frames_dir, chunks, video_path, index_path)

memory_after = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory usage: {memory_after - memory_before:.1f} MB")
```

## Convenience Functions

### Quick Assembly

```python
from modules.video_assembler import assemble_video_from_frames

# Simple assembly from frames
result = assemble_video_from_frames(
    frames_dir="./frames",
    chunks_data=chunk_data,
    output_video="./library.mp4",
    output_index="./library_index.json"
)
```

## Dependencies

- **memvid**: Core video assembly functionality
- **json**: Index file creation and validation
- **pathlib**: Path handling
- **typing**: Type hints
- **contextlib**: Output suppression
- **warnings**: Warning management