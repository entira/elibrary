# eLibrary - PDF Knowledge Base with RAG

Advanced PDF processing system that converts document libraries into searchable video-based indexes using Memvid technology and Retrieval Augmented Generation (RAG).

## 🏗️ System Architecture

Two-version processing pipeline with enhanced metadata extraction and cross-page context preservation:

- **V1 Processor**: Basic PDF chunking (512 chars, ~480 avg length)
- **V2 Processor**: Enhanced chunking (400 chars, ~362 avg length) with detailed page metadata

## 🎯 Features

- **Enhanced PDF Processing**: Two-tier processing system (V1 basic, V2 enhanced)
- **AI Metadata Extraction**: Ollama-powered extraction of titles, authors, publishers, years
- **Video-based Indexing**: QR-encoded text chunks in video frames for efficient storage
- **Cross-page Context**: Enhanced V2 processor preserves context between pages
- **RAG Integration**: Ready for Retrieval Augmented Generation workflows
- **Interactive Chat**: Query your PDF library with natural language

## 📋 Požiadavky

### Systémové požiadavky
- Python >= 3.10
- Ollama server bežiaci na `localhost:11434`
- Modely v Ollama: `mistral:latest` a `nomic-embed-text`

### Python závislosti
```bash
pip install -r requirements.txt
```

```txt
memvid
PyPDF2
requests
tqdm
```

## 🏗️ Architektúra

### Core Components

#### 1. **OllamaEmbedder**
- **Purpose**: Generate embeddings using `nomic-embed-text` model
- **Endpoint**: `POST http://localhost:11434/api/generate`
- **Parameters**: `{"model": "nomic-embed-text", "embedding": true}`

#### 2. **PDFLibraryProcessor (V1)**
- **Basic processing** with standard chunking
- **Configuration**:
  - Input folder: `./pdf_books`
  - Output folder: `./memvid_out`
  - Chunk size: 512 characters, Overlap: 50 characters
  - Output: 8,975 segments, ~482 chars average

#### 3. **PDFLibraryProcessorV2 (Enhanced)**
- **Advanced processing** with detailed page metadata
- **Configuration**:
  - Input folder: `./pdf_books`
  - Output folder: `./memvid_out_v2`
  - Chunk size: 400 characters, Overlap: 50 characters
  - Output: 14,486 segments, ~362 chars average
  - Features: Cross-page chunks, detailed page references

### Workflow

```mermaid
graph TD
    A[PDF súbory] --> B[Extrakcia textu]
    B --> C[Chunking ~100 slov]
    C --> D[Prvých 10 chunks]
    D --> E[Ollama mistral:latest]
    E --> F[JSON metadáta]
    F --> G[Memvid encoder]
    G --> H[QR frames generovanie]
    H --> I[Video + Index]
```

## 🔧 Funkcionalita

### 1. Extrakcia metadát pomocou AI

```python
def extract_metadata_with_ollama(self, sample_text: str) -> Dict[str, str]:
```

**Proces**:
1. Vezme prvých 10 chunks z PDF
2. Spojí ich do sample_text 
3. Pošle prompt do `mistral:latest`:
   ```
   Extract JSON with keys: title, authors, publishers, year, doi from this text:
   [sample_text]
   Return only valid JSON.
   ```
4. Parsuje JSON odpoveď
5. Validuje a normalizuje dáta

**Výstup**:
```json
{
  "title": "RAG-Driven Generative AI",
  "authors": "Denis Rothman",
  "publishers": "Packt Publishing", 
  "year": "2024",
  "doi": "10.1234/example"
}
```

### 2. PDF Processing Pipeline

```python
def process_pdf(self, pdf_path: Path) -> bool:
```

**Kroky**:
1. **Extrakcia textu** pomocou PyPDF2
2. **Chunking** na ~100 slov per chunk
3. **Metadata extraction** z prvých 10 chunks
4. **Pridanie do memvid** encoder

### 3. Video Index Generation

```python
encoder.add_pdf(str(pdf_path), chunk_size=512, overlap=50)
encoder.build_video("library.mp4", "library_index.json")
```

**Vytvorí**:
- 🎥 **library.mp4**: Video kde každý frame obsahuje QR kód s chunks
- 📋 **library_index.json**: Metadata a mappings
- 🔍 **library_index.faiss**: Vector search index

## 📁 Project Structure

```
eLibrary/
├── pdf_library_processor.py      # V1: Basic PDF processor
├── pdf_library_processor_v2.py   # V2: Enhanced processor with page metadata
├── pdf_library_chat.py           # Interactive chat interface
├── requirements.txt              # Python dependencies
├── README.md                     # This documentation
├── .gitignore                    # Git ignore rules
├── .github/                      # 🤖 GitHub Actions automation
│   ├── workflows/
│   │   └── auto-fix-issues.yml   # Automated issue fixing workflow
│   ├── actions/
│   │   └── claude-ai-fix/        # Custom AI fix action
│   ├── ISSUE_TEMPLATE/           # Structured issue templates
│   │   ├── auto-fix-bug.md       # Bug report template
│   │   └── auto-fix-enhancement.md # Enhancement template
│   ├── README.md                 # GitHub Actions documentation
│   ├── SETUP_INSTRUCTIONS.md     # Setup guide
│   └── test_auto_fix.py          # Test script
├── pdf_books/                    # Input PDF files (excluded from git)
│   ├── book1.pdf
│   └── book2.pdf
├── memvid_out/                   # V1 Output (excluded from git)
│   ├── library.mp4              # V1 Video index
│   ├── library_index.json       # V1 Metadata (8,975 segments)
│   └── library_index.faiss      # V1 Vector index
├── memvid_out_v2/               # V2 Enhanced Output (excluded from git)
│   ├── library_v2.mp4           # V2 Video index
│   ├── library_v2_index.json    # V2 Enhanced metadata (14,486 segments)
│   └── library_v2_index.faiss   # V2 Vector index
└── venv/                        # Python virtual environment (excluded)
```

## 🚀 Použitie

### 1. Príprava prostredia

```bash
# Vytvorenie virtual environment
python3 -m venv venv
source venv/bin/activate

# Inštalácia závislostí
pip install -r requirements.txt
```

### 2. Spustenie Ollama

```bash
# Stiahnutie modelov
ollama pull mistral:latest
ollama pull nomic-embed-text

# Overenie že Ollama beží
curl http://localhost:11434/api/tags
```

### 3. Pridanie PDF súborov

```bash
# Vytvorenie priečinka a pridanie PDF
mkdir -p pdf_books
cp *.pdf pdf_books/
```

### 4. Run Processing

**Option A: Basic V1 Processor**
```bash
python3 pdf_library_processor.py
```

**Option B: Enhanced V2 Processor (Recommended)**
```bash
python3 pdf_library_processor_v2.py
```

**V2 Benefits:**
- 61% more text segments (14,486 vs 8,975)
- Shorter, more precise chunks (362 vs 482 chars)
- Cross-page context preservation (2,184 cross-page chunks)
- Detailed page metadata for each chunk
- Enhanced statistics and book information

### 5. Chat s knižnicou

Po vytvorení video indexu môžete spustiť chat interface:

```bash
python3 pdf_library_chat.py
```

## 📊 Processing Results

### V2 Enhanced Output (Recommended)
```
Found 7 PDF files to process
Output directory: memvid_out_v2
Chunk size: 400 chars, Overlap: 50 chars

Processing: RAG-Driven Generative AI...
  - Pages: 517, Enhanced chunks: 2316
  - Extracting metadata...
  - Title: RAG-Driven Generative AI
  - Authors: Denis Rothman
  - Year: 2024

Generating QR frames: 100%|██████| 14486/14486 [24:09<00:00]
Enhancing index with detailed metadata...

✅ SUCCESS!
📚 Processed 7 PDF books
🎥 Enhanced video: memvid_out_v2/library_v2.mp4
📋 Enhanced index: memvid_out_v2/library_v2_index.json
📄 Each chunk includes detailed page references!
```

### Performance Comparison

| Metric | V1 Basic | V2 Enhanced | Improvement |
|--------|----------|-------------|-------------|
| Total segments | 8,975 | 14,486 | +61% |
| Avg segment length | 482 chars | 362 chars | -25% (more precise) |
| Cross-page chunks | 0 | 2,184 | New feature |
| Page metadata | Basic | Detailed | Enhanced |
| Unique pages indexed | N/A | 2,245 | New feature |

### Vytvorené súbory

#### `library.mp4`
- Video kde každý frame = 1 chunk ako QR kód
- Framerate: 1 FPS (default memvid)
- Formát: MP4 s H.264 codec

#### `library_v2_index.json` (Enhanced)
```json
{
  "metadata": [
    {
      "id": 0,
      "text": "RAG-Driven Generative AI\nBuild custom retrieval...",
      "frame": 0,
      "length": 134
    }
  ],
  "enhanced_stats": {
    "total_files": 7,
    "total_chunks": 14486,
    "total_unique_pages": 2245,
    "cross_page_chunks": 2184,
    "files": {
      "RAG-Driven Generative AI...pdf": {
        "chunks": 2316,
        "unique_pages": 517,
        "title": "RAG-Driven Generative AI",
        "authors": "Denis Rothman",
        "year": "2024"
      }
    }
  }
}
```

#### `library_index.faiss`
- Binárny FAISS vector index
- 384-dimenzionálne embeddings (nomic-embed-text)
- Umožňuje semantické vyhľadávanie

## 🔍 Interné detaily

### Chunk Storage Lifecycle

1. **V pamäti**: `encoder.chunks[]` - list všetkých chunks
2. **Dočasne**: `/tmp/tmp*/frames/frame_*.png` - QR obrázky
3. **Finálne**: `library.mp4` + index súbory

### Error Handling

```python
# PDF parsing errors
except Exception as e:
    print(f"Error reading PDF {pdf_path}: {e}")
    return [], 0

# Ollama API errors  
except Exception as e:
    print(f"Error extracting metadata with Ollama: {e}")
    return self._empty_metadata()
```

### Performance

- **Chunk generovanie**: ~10 chunks/sec
- **QR frame generovanie**: ~10-15 frames/sec  
- **Metadata extrakcia**: ~5-10 sec per PDF
- **Celkový čas**: ~2-3 min pre 7 PDF (varies by size)

## 🎛️ Konfigurácia

### Memvid parametre
```python
# V PDFLibraryProcessor.__init__()
self.encoder = MemvidEncoder()

# V process_pdf()
self.encoder.add_pdf(str(pdf_path), chunk_size=512, overlap=50)
```

### Ollama endpointy
```python
# Embeddings
POST http://localhost:11434/api/generate
{
  "model": "nomic-embed-text",
  "prompt": text,
  "embedding": true
}

# Metadata extraction
POST http://localhost:11434/api/generate  
{
  "model": "mistral:latest",
  "prompt": "Extract JSON...",
  "options": {"temperature": 0.1}
}
```

## ⚠️ Známe limitácie

1. **PDF parsing**: Niektoré PDF môžu mať problémy s text extraction
2. **Ollama dostupnosť**: Vyžaduje bežiaci Ollama server
3. **Memory usage**: Veľké PDF môžu spotrebovať veľa RAM
4. **Processing time**: Video generovanie je časovo náročné
5. **Metadata quality**: Závisí od kvality text extraction a AI modelu

## 🔧 Troubleshooting

### Ollama connection failed
```bash
# Skontrolovať či Ollama beží
ollama list
curl http://localhost:11434/api/tags
```

### PDF extraction errors
```python
# Skontrolovať PyPDF2 log v konzole
# Niektoré PDF môžu byť chránené/poškodené
```

### Memory errors
```bash
# Spracovávať PDF po menších dávkach
# Alebo zvýšiť system memory/swap
```

## 🤖 GitHub Actions Automatizácia

✅ **Kompletne implementovaný automatizovaný systém na riešenie issues!**

### 🎯 Ako používať:

#### 1. **Vytvorte issue s template:**
```
GitHub → Issues → New issue → Choose template:
- 🤖 Auto-Fix Bug Report
- 🚀 Auto-Fix Enhancement
```

#### 2. **Aktivujte automatizáciu:**
```bash
# Pridajte label na issue
gh issue edit ISSUE_NUMBER --add-label "auto-fix"

# Alebo cez web interface v GitHub
```

#### 3. **Sledujte progress:**
```bash
# GitHub Actions workflow sa automaticky spustí
https://github.com/entira/elibrary/actions

# AI vytvorí PR s riešením
gh pr list --label "auto-fix"
```

### 🛠️ Implementované komponenty:
- ✅ **GitHub Actions workflow** (`.github/workflows/auto-fix-issues.yml`)
- ✅ **Issue templates** s štruktúrovaným formulárom
- ✅ **Custom AI action** pre automated fixes
- ✅ **Automated testing** a validation
- ✅ **Smart labeling system**

### 📋 Ako to funguje:
1. **Issue detection** - workflow reaguje na label `auto-fix`
2. **AI analysis** - analyzuje problém a kód context
3. **Fix generation** - vytvorí riešenie based on issue description
4. **Automated testing** - validuje syntax a imports
5. **PR creation** - vytvorí pull request s fix
6. **Manual review** - vyžaduje human approval pred merge

### 📚 Dokumentácia:
- **Setup guide**: [`.github/SETUP_INSTRUCTIONS.md`](.github/SETUP_INSTRUCTIONS.md)
- **System overview**: [`.github/README.md`](.github/README.md)
- **Test script**: [`.github/test_auto_fix.py`](.github/test_auto_fix.py)

## 📈 Rozšírenia

### Možné vylepšenia
1. **Batch processing**: Spracovanie po dávkach pre veľké kolekcie
2. **Multi-threading**: Paralelné spracovanie PDF
3. **Database storage**: Ukladanie do DB namiesto JSON
4. **Web interface**: GUI pre browsing a vyhľadávanie
5. **Alternative models**: Podpora pre iné LLM/embedding modely

### Custom metadata fields
```python
# Pridať nové pole do metadata extraction prompt
prompt = f"""Extract JSON with keys: title, authors, publishers, year, doi, isbn, language from this text:
{sample_text}
Return only valid JSON."""
```

# PDF Library Chat Interface

Interaktívny chat systém pre komunikáciu s PDF knižnicou pomocou video pamäte.

## 🎯 Funkcionalita

### Chat Commands
```
help          - Zobrazí nápovedu
info          - Informácie o knižnici  
search <query>- Vyhľadávanie v obsahu
stats         - Štatistiky session
clear         - Vyčistí obrazovku
exit/quit     - Ukončí chat
```

### Príklady používania

```bash
🤔 You: What is RAG in AI?
🤖 Assistant: Based on the library content, RAG (Retrieval Augmented Generation) is...

🤔 You: search machine learning
🔍 Search results for: 'machine learning' (0.15s)
📄 Relevant passages:
──────────────────────────────────────────────────
[Relevant text chunks from PDFs...]
──────────────────────────────────────────────────

🤔 You: info
📖 Library Overview:
   📚 Total books: 7
   📝 Total chunks: 8975

📑 Books in library:
   1. RAG-Driven Generative AI
      📖 Author(s): Denis Rothman
      📅 Year: 2024
      📝 Chunks: 1079
```

## 🔧 Technické detaily

### Komponenty
- **PDFLibraryChat**: Hlavná trieda pre chat interface
- **OllamaLLM**: Lokálne LLM pre generovanie odpovedí  
- **MemvidChat**: Video pamäť search a retrieval

### Workflow
1. **Načítanie video indexu** a validácia súborov
2. **Semantic search** v PDF chunks pomocou embeddings
3. **Context retrieval** z relevantných chunks
4. **LLM response** pomocou Ollama mistral:latest
5. **Formátovaný výstup** s metadátami

### Konfigurácia
```python
# V PDFLibraryChat.__init__()
self.chat = MemvidChat(video_file, index_file)
self.llm = OllamaLLM(model="mistral:latest")
```

## 📝 Version History

### V2.0 (Current) - Enhanced Processing
- **14,486 segments** with detailed page metadata
- **Cross-page chunks** for better context preservation
- **Enhanced statistics** and book information
- **Improved chunking** (400 chars vs 512 chars)
- **Better RAG performance** with more precise segments

### V1.0 - Basic Processing
- **8,975 segments** with basic metadata
- **512-character chunks** with standard overlap
- **Simple PDF processing** without page references

---

**Repository**: eLibrary PDF Knowledge Base  
**Version**: 2.0 Enhanced  
**Last Updated**: December 2024  
**License**: MIT