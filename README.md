# eLibrary - PDF Knowledge Base with RAG

Advanced PDF processing system that converts document libraries into searchable video-based indexes using Memvid technology and Retrieval Augmented Generation (RAG).

## Quick Start

### Prerequisites

- Python 3.8+
- Ollama with mistral:latest model
- MemVid library

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/entira/elibrary.git
cd elibrary
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup Ollama**
```bash
# Install Ollama from https://ollama.ai
ollama pull mistral:latest
ollama pull nomic-embed-text
```

4. **Add PDF files**
```bash
# Place your PDF files in the pdf_books directory
mkdir -p pdf_books
# Copy your PDF files here
```

### Basic Usage

1. **Process PDF library**
```bash
python3 pdf_library_processor.py
```

2. **Chat with your documents**
```bash
python3 pdf_chat.py
```

## Features

### Current Capabilities

- **Token-based Chunking**: Smart sliding window chunking (500 tokens, 15% overlap)
- **Enhanced Metadata**: Automatic extraction of titles, authors, publishers, publication years
- **Page-accurate Citations**: Precise page references in chat responses
- **Cross-page Context**: Context preservation across page boundaries
- **Smart Skip Processing**: Avoids reprocessing already processed PDFs
- **Parallel Processing**: Multi-worker QR generation for faster processing
- **Clean Output**: Comprehensive warning suppression for professional experience

### Architecture

```mermaid
graph LR
    subgraph "Input"
        PDF[📄 PDF Files]
    end
    
    subgraph "Processing"
        EXTRACT[🔧 Text Extraction]
        CHUNK[📝 Token Chunking]
        META[📊 Metadata Extraction]
        QR[📱 QR Generation]
        VIDEO[🎥 Video Assembly]
    end
    
    subgraph "Output"
        VID[🎥 library.mp4]
        INDEX[📋 library_index.json]
    end
    
    PDF --> EXTRACT
    EXTRACT --> CHUNK
    CHUNK --> META
    META --> QR
    QR --> VIDEO
    VIDEO --> VID
    VIDEO --> INDEX
```

### File Structure

```
elibrary/
├── pdf_books/                   # Input PDFs
├── pdf_library_processor.py     # Main processor
├── pdf_chat.py                  # Chat interface
├── requirements.txt             # Dependencies
├── memvid_out/                  # Output (excluded from git)
│   ├── library.mp4              # Video index
│   ├── library_index.json       # Enhanced metadata
│   └── library_index.faiss      # Vector index
└── README.md
```

## Usage Examples

### Processing Options

**Standard processing:**
```bash
python3 pdf_library_processor.py
```

**Force reprocess all files:**
```bash
python3 pdf_library_processor.py --force-reprocess
```

**Use multiple workers:**
```bash
python3 pdf_library_processor.py --max-workers 8
```

### Chat Interface

The chat system provides:
- Interactive Q&A with your PDF library
- Automatic citations with book titles and page numbers
- Context-aware responses using RAG
- Real-time search across all processed documents

Example interaction:
```
> What are the best practices for podcasting?

Based on your library, here are key podcasting best practices:

1. Consistent publishing schedule is crucial for audience retention. [Podcasting 100 Success Secrets, page 23]

2. Quality audio equipment significantly impacts listener experience. [Podcasting for Dummies, page 45]

3. Engaging content should balance entertainment with valuable information. [Profitable Podcasting, page 67]
```

## Configuration

### Environment Setup

The system automatically detects and uses:
- Local Ollama installation at `http://localhost:11434`
- PDF files in `./pdf_books/` directory
- Output directory at `./memvid_out/`

### Dependencies

Core dependencies (see `requirements.txt`):
- `memvid` - Video indexing library
- `pymupdf` - PDF text extraction
- `tiktoken` - Token-based chunking
- `requests` - HTTP communication
- `tqdm` - Progress tracking

## Technical Details

### Processing Pipeline

1. **PDF Text Extraction**: Uses PyMuPDF for high-quality text extraction
2. **Text Cleaning**: Removes encoding artifacts and formatting issues
3. **Token-based Chunking**: Creates 500-token chunks with 15% overlap
4. **Metadata Extraction**: Uses LLM to extract structured metadata
5. **QR Code Generation**: Parallel generation of video frames
6. **Index Building**: Creates searchable vector index

### Chunking Strategy

- **Chunk Size**: 500 tokens (~2000 characters)
- **Overlap**: 15% (75 tokens) for context continuity
- **Boundary Detection**: Smart sentence boundary preservation
- **Cross-page Chunks**: Additional chunks spanning page boundaries

### Performance Optimizations

- **Smart Skip**: Avoids reprocessing existing files
- **Parallel QR Generation**: Multi-worker processing
- **Warning Suppression**: Clean output without library warnings
- **Progress Tracking**: Real-time processing feedback

## Troubleshooting

### Common Issues

**Ollama Connection**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

**Missing Models**
```bash
# Pull required models
ollama pull mistral:latest
ollama pull nomic-embed-text
```

**Memory Issues**
```bash
# Reduce workers for low-memory systems
python3 pdf_library_processor.py --max-workers 2
```

## Future Roadmap

### Planned Features

- **CDN/S3 Streaming**: On-demand video frame streaming from cloud storage
- **Content Encryption**: AES-256-GCM encryption for QR code content
- **Web3 Integration**: Crypto wallet-based key derivation for access control
- **Anonymous Access**: Tor hidden service integration
- **MCP Server**: Model Context Protocol implementation for AI assistant integration
- **Advanced Search**: Semantic search with similarity scoring
- **Multi-format Support**: EPUB, DOCX, and other document formats

### Architecture Evolution

Future versions will support:
- Distributed processing across multiple nodes
- Real-time collaborative editing
- Integration with external knowledge bases
- Advanced analytics and usage metrics
- Mobile application support

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. See LICENSE file for details.