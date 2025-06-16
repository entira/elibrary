# Multi-Library Support Testing

This document describes testing procedures for the new multi-library functionality.

## Test Setup

### Directory Structure
```
library/
├── 1/                  # First library instance
│   ├── pdf/           # PDFs for library 1
│   └── data/          # Generated files (if processed)
├── 2/                 # Second library instance
│   ├── pdf/           # PDFs for library 2
│   └── data/          # Generated files (if processed)
└── 3/                 # Third library instance (etc.)
    └── pdf/           # Empty = ignored by discovery
```

### Test Commands

1. **Test Library Discovery**
```bash
python3 -c "
from pdf_library_processor import PDFLibraryProcessorV2
processor = PDFLibraryProcessorV2()
libraries = processor.discover_libraries()
for lib in libraries:
    print(f'Library {lib[\"id\"]}: {lib[\"pdf_count\"]} PDFs, processed: {lib[\"is_processed\"]}')
"
```

2. **Test Processing Multiple Libraries**
```bash
# Process all discovered libraries
python3 pdf_library_processor.py

# Force reprocess all libraries  
python3 pdf_library_processor.py --force-reprocess

# Use custom library root
python3 pdf_library_processor.py --library-root ./my_libraries
```

3. **Test Multi-Library Chat**
```bash
python3 pdf_chat.py
```

## Expected Behavior

### Processor Behavior
- Discovers all `library/[1,2,3,...]/pdf/` directories with PDF files
- Skips libraries that already have `data/library.mp4` and `data/library_index.json`
- Processes only libraries that need processing
- Shows comprehensive progress and statistics
- Creates separate video/index files for each library

### Chat Behavior  
- Loads all available libraries automatically
- Searches across all libraries simultaneously
- Shows library source in citations: `[Book Title, page X - Library N]`
- Displays combined statistics from all libraries
- No library selection menu (automatic multi-library mode)

## Test Scenarios

### Scenario 1: Fresh Setup
1. Create `library/1/pdf/` and add PDFs
2. Run processor → should process library 1
3. Create `library/2/pdf/` and add different PDFs  
4. Run processor again → should skip library 1, process library 2
5. Run chat → should search both libraries

### Scenario 2: Skip Mechanism
1. Have processed library 1
2. Run processor → should skip library 1
3. Use `--force-reprocess` → should reprocess library 1

### Scenario 3: Multi-Library Chat
1. Have 2+ processed libraries
2. Run chat → should show combined statistics
3. Search for content → should return results from all libraries
4. Check citations → should include library source

## Known Limitations
- Library directories must be numeric (1, 2, 3, ...)
- Each library processes independently (no cross-library deduplication)
- Chat loads all libraries in memory simultaneously
- No library-specific search filtering in chat interface

## Development Notes
- Branch: `multi-library-support`
- Modified files: `pdf_library_processor.py`, `pdf_chat.py`
- New classes: `MultiLibraryRetriever`
- Skip mechanism integrated with existing index-based detection