#!/usr/bin/env python3
"""
TextExtractor Module

Handles PDF text extraction with page-by-page processing, text cleaning,
and page offset detection for optimal metadata extraction.
"""

import re
import warnings
from pathlib import Path
from typing import Dict, Tuple
import pymupdf as fitz

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


class TextExtractor:
    """PDF text extraction with advanced cleaning and page analysis."""
    
    def __init__(self):
        """Initialize TextExtractor."""
        pass
    
    def extract_text_with_pages(self, pdf_path: Path) -> Tuple[Dict[int, str], int, int]:
        """Extract text from PDF with page-by-page mapping and detect page offset.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (page_texts_dict, num_pages, page_offset)
            - page_texts_dict: Dict mapping page numbers to cleaned text
            - num_pages: Total number of pages in PDF
            - page_offset: Detected page numbering offset
        """
        try:
            # Open PDF with PyMuPDF using context manager for automatic cleanup
            with fitz.open(pdf_path) as doc:
                num_pages = len(doc)
                page_texts = {}

                for page_num in range(num_pages):
                    try:
                        page = doc[page_num]
                        # Extract text from page
                        page_text = page.get_text()
                        # Clean the extracted text to fix Issue #2 problems
                        cleaned_text = self.clean_extracted_text(page_text)
                        # Use actual PDF page numbers (0-based index + 1 = real page number)
                        actual_page_num = page_num + 1
                        page_texts[actual_page_num] = cleaned_text
                    except Exception as e:
                        actual_page_num = page_num + 1
                        print(f"Error extracting text from page {actual_page_num}: {e}")
                        page_texts[actual_page_num] = ""
            
            # Detect page numbering offset
            page_offset = self.detect_page_number_offset(page_texts)
            
            return page_texts, num_pages, page_offset
                
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return {}, 0, 0
    
    def extract_first_page_text(self, pdf_path: Path) -> str:
        """Extract text from the first page of PDF for metadata extraction.
        If first page is empty (image-based), try next few pages.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Cleaned text from first available page
        """
        try:
            with fitz.open(pdf_path) as doc:
                if len(doc) == 0:
                    return ""

                # Try first few pages to find one with extractable text
                for page_num in range(min(3, len(doc))):
                    page = doc[page_num]
                    page_text = page.get_text()

                    if page_text and page_text.strip():
                        # Use simple cleaning to avoid aggressive removal
                        cleaned_text = page_text.strip().replace('\\x00', '')
                        return cleaned_text

                return ""
            
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def clean_extracted_text(self, text: str) -> str:
        """Clean extracted PDF text from encoding issues and artifacts.
        
        Args:
            text: Raw text extracted from PDF
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Remove null bytes and other control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        # Fix multiple spaces while preserving intentional formatting
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'[ \t]*\n[ \t]*', '\n', text)  # Clean line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit multiple newlines
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])(\s+)([A-Z])', r'\1 \3', text)  # Fix split words like "P ackt"
        text = re.sub(r'([a-z])\s+([a-z])\s+([a-z])', lambda m: m.group(0) if len(m.group(0)) > 10 else m.group(1) + m.group(2) + m.group(3), text)
        
        # Remove artifacts and clean up
        text = text.replace('\u0000', '')  # Remove null characters
        text = text.replace('\ufffd', '')  # Remove replacement characters
        text = text.replace('\x00', '')   # Remove more null variants
        
        # Remove special tokens that can cause tokenizer issues
        text = text.replace('<|endoftext|>', '')
        text = text.replace('<|startoftext|>', '')
        text = text.replace('<|im_start|>', '')
        text = text.replace('<|im_end|>', '')
        
        # Fix weird spacing patterns common in academic PDFs
        text = re.sub(r'([a-z])\s+([a-z])(?=\s)', r'\1\2', text)  # Fix "w o r d" -> "word"
        text = re.sub(r'\b([A-Z])\s+([a-z])', r'\1\2', text)  # Fix "T ext" -> "Text"
        
        # Clean up excessive whitespace while preserving paragraph structure
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n[ \t]+', '\n', text)  # Remove leading whitespace on lines
        text = re.sub(r'[ \t]+\n', '\n', text)  # Remove trailing whitespace on lines
        
        # Final cleanup
        text = text.strip()
        
        return text
    
    def detect_page_number_offset(self, page_texts: Dict[int, str]) -> int:
        """Detect page numbering offset by analyzing page content.
        
        Args:
            page_texts: Dictionary mapping page numbers to text content
            
        Returns:
            Detected offset (0 if no clear pattern found)
        """
        try:
            # Look for explicit page numbers in text
            for page_num, text in page_texts.items():
                if not text.strip():
                    continue
                
                # Look for page numbers at start or end of text
                lines = text.split('\n')
                
                # Check first few and last few lines for page numbers
                check_lines = lines[:3] + lines[-3:] if len(lines) > 6 else lines
                
                for line in check_lines:
                    line = line.strip()
                    
                    # Look for standalone numbers that might be page numbers
                    page_num_match = re.search(r'^\s*(\d+)\s*$', line)
                    if page_num_match:
                        detected_page = int(page_num_match.group(1))
                        # Calculate offset (actual page number - detected page number)
                        offset = page_num - detected_page
                        if 0 <= offset <= 10:  # Reasonable offset range
                            return offset
                
                # Look for "Page X" patterns
                page_pattern_match = re.search(r'(?i)page\s+(\d+)', text)
                if page_pattern_match:
                    detected_page = int(page_pattern_match.group(1))
                    offset = page_num - detected_page
                    if 0 <= offset <= 10:
                        return offset
            
            # No clear pattern found, assume no offset
            return 0
            
        except Exception as e:
            print(f"Error detecting page offset: {e}")
            return 0


# Utility functions for standalone usage
def extract_pdf_text(pdf_path: str) -> Dict[int, str]:
    """Convenience function to extract text from PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary mapping page numbers to text content
    """
    extractor = TextExtractor()
    page_texts, _, _ = extractor.extract_text_with_pages(Path(pdf_path))
    return page_texts


def extract_first_page(pdf_path: str) -> str:
    """Convenience function to extract first page text.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Text content from first available page
    """
    extractor = TextExtractor()
    return extractor.extract_first_page_text(Path(pdf_path))


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python text_extractor.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    print(f"Extracting text from: {pdf_path}")
    
    extractor = TextExtractor()
    page_texts, num_pages, offset = extractor.extract_text_with_pages(Path(pdf_path))
    
    print(f"📄 Extracted {num_pages} pages")
    print(f"🔢 Page offset: {offset}")
    print(f"📝 Total characters: {sum(len(text) for text in page_texts.values())}")
    
    # Show preview of first page
    if page_texts:
        first_page = min(page_texts.keys())
        preview = page_texts[first_page][:300]
        print(f"\\n📖 First page preview:\\n{preview}...")