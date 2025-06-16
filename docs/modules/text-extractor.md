# TextExtractor Module

## Overview

The `TextExtractor` module handles PDF text extraction with advanced page-by-page processing, intelligent text cleaning, and page offset detection for optimal metadata extraction.

## Features

- **Multi-page text extraction** with page-by-page mapping
- **Intelligent first page extraction** for metadata purposes
- **Advanced text cleaning** to fix PDF extraction artifacts
- **Page offset detection** for accurate citation references
- **Error handling** for corrupted or image-based PDFs

## Installation

```bash
pip install pymupdf
```

## Basic Usage

### Extract All Pages

```python
from modules.text_extractor import TextExtractor

extractor = TextExtractor()
page_texts, num_pages, offset = extractor.extract_text_with_pages("document.pdf")

print(f"Extracted {num_pages} pages with offset {offset}")
for page_num, text in page_texts.items():
    print(f"Page {page_num}: {len(text)} characters")
```

### Extract First Page Only

```python
from modules.text_extractor import TextExtractor

extractor = TextExtractor()
first_page_text = extractor.extract_first_page_text("document.pdf")
print(f"First page: {len(first_page_text)} characters")
```

### Convenience Functions

```python
from modules.text_extractor import extract_pdf_text, extract_first_page

# Extract all pages
page_texts = extract_pdf_text("document.pdf")

# Extract first page
first_page = extract_first_page("document.pdf")
```

## API Reference

### TextExtractor Class

#### `__init__()`

Initialize the TextExtractor.

#### `extract_text_with_pages(pdf_path: Path) -> Tuple[Dict[int, str], int, int]`

Extract text from PDF with page-by-page mapping and detect page offset.

**Parameters:**
- `pdf_path`: Path to PDF file

**Returns:**
- `page_texts_dict`: Dictionary mapping page numbers to cleaned text
- `num_pages`: Total number of pages in PDF
- `page_offset`: Detected page numbering offset

#### `extract_first_page_text(pdf_path: Path) -> str`

Extract text from the first page of PDF for metadata extraction.
If first page is empty (image-based), tries next few pages.

**Parameters:**
- `pdf_path`: Path to PDF file

**Returns:**
- Cleaned text from first available page

#### `clean_extracted_text(text: str) -> str`

Clean extracted PDF text from encoding issues and artifacts.

**Parameters:**
- `text`: Raw text extracted from PDF

**Returns:**
- Cleaned and normalized text

#### `detect_page_number_offset(page_texts: Dict[int, str]) -> int`

Detect page numbering offset by analyzing page content.

**Parameters:**
- `page_texts`: Dictionary mapping page numbers to text content

**Returns:**
- Detected offset (0 if no clear pattern found)

## Text Cleaning Features

The module automatically handles common PDF extraction issues:

- **Control characters removal**: Removes null bytes and control characters
- **Spacing normalization**: Fixes multiple spaces and line breaks
- **Word splitting repair**: Fixes split words like "P ackt" → "Packet"
- **Academic PDF formatting**: Handles common formatting in academic papers
- **Encoding issues**: Removes replacement characters and encoding artifacts

## Page Offset Detection

The module automatically detects page numbering offsets by looking for:

- Standalone page numbers in text
- "Page X" patterns
- Consistent numbering patterns across pages

This ensures accurate citation references even when PDFs have front matter or different numbering schemes.

## Error Handling

### Graceful Degradation

- **Empty pages**: Skipped automatically
- **Extraction failures**: Individual page failures don't stop processing
- **Corrupted PDFs**: Returns empty results with error messages
- **Image-based PDFs**: Tries multiple pages to find extractable text

### Error Messages

All errors are logged with descriptive messages:

```python
# Example error handling
try:
    page_texts, num_pages, offset = extractor.extract_text_with_pages(pdf_path)
except Exception as e:
    print(f"Extraction failed: {e}")
```

## Performance Considerations

### Memory Usage

- Processes pages individually to minimize memory usage
- Closes PDF documents promptly
- Efficient text cleaning with compiled regex patterns

### Processing Speed

- Fast PyMuPDF backend
- Optimized text cleaning algorithms
- Page offset detection with early termination

## Command Line Usage

```bash
# Extract text from PDF
python text_extractor.py document.pdf

# Example output:
# 📄 Extracted 150 pages
# 🔢 Page offset: 2
# 📝 Total characters: 125,432
# 📖 First page preview: This is a sample document...
```

## Integration Examples

### With Metadata Extraction

```python
from modules.text_extractor import TextExtractor
from modules.metadata_extractor import MetadataExtractor

# Extract first page for metadata
extractor = TextExtractor()
first_page = extractor.extract_first_page_text(pdf_path)

# Extract metadata
metadata_extractor = MetadataExtractor()
metadata = metadata_extractor.extract_metadata(first_page, pdf_path.name)
```

### With Text Chunking

```python
from modules.text_extractor import TextExtractor
from modules.text_chunker import TextChunker

# Extract all pages
extractor = TextExtractor()
page_texts, num_pages, offset = extractor.extract_text_with_pages(pdf_path)

# Create chunks
chunker = TextChunker()
chunks = chunker.create_enhanced_chunks(page_texts, offset)
```

## Troubleshooting

### Common Issues

**No text extracted:**
- PDF might be image-based (scanned document)
- Try OCR preprocessing
- Check if PDF is corrupted

**Garbled text:**
- PDF has unusual encoding
- Text cleaning should handle most cases
- May need custom cleaning rules

**Wrong page offset:**
- PDF has unusual numbering scheme
- Manual offset can be provided to chunker
- Check page content for numbering patterns

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

extractor = TextExtractor()
result = extractor.extract_text_with_pages(pdf_path)
```

## Dependencies

- **pymupdf**: PDF text extraction
- **pathlib**: Path handling
- **re**: Text cleaning regex operations
- **typing**: Type hints