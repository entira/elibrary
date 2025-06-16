#!/usr/bin/env python3
"""
Test script for improved metadata extraction
"""

import warnings
import os
import sys
from io import StringIO
import contextlib
import json
import re
import requests
from pathlib import Path
import pymupdf as fitz
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONWARNINGS'] = 'ignore'

def suppress_stdout():
    return contextlib.redirect_stdout(StringIO())

def suppress_stderr():  
    return contextlib.redirect_stderr(StringIO())


class ImprovedMetadataExtractor:
    """Improved metadata extractor with validation and fallbacks."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        
    def clean_extracted_text(self, text: str) -> str:
        """Clean extracted PDF text from encoding issues and artifacts."""
        if not text:
            return ""
        
        # Remove null bytes and other control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        # Fix multiple spaces while preserving intentional formatting
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'[ \t]*\n[ \t]*', '\n', text)  # Clean line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit multiple newlines
        
        return text.strip()
    
    def extract_text_sample(self, pdf_path: Path, pages: int = 3) -> str:
        """Extract text sample from first few pages of PDF."""
        try:
            doc = fitz.open(pdf_path)
            sample_text = ""
            
            for page_num in range(min(pages, len(doc))):
                page = doc[page_num]
                page_text = page.get_text()
                cleaned_text = self.clean_extracted_text(page_text)
                sample_text += cleaned_text + "\n\n"
                
                # Stop if we have enough text
                if len(sample_text) > 3000:
                    break
            
            doc.close()
            return sample_text.strip()
            
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """Fallback: extract metadata from filename patterns."""
        metadata = {
            "title": "",
            "authors": "",
            "publishers": "",
            "year": "",
            "doi": ""
        }
        
        # Common filename patterns: "Title -- Author -- Year -- Publisher -- ISBN -- Source.pdf"
        filename_clean = filename.replace('.pdf', '')
        
        # Try to parse structured filenames
        if ' -- ' in filename_clean:
            parts = filename_clean.split(' -- ')
            if len(parts) >= 4:
                metadata["title"] = parts[0].strip()
                metadata["authors"] = parts[1].strip() if parts[1].strip() != '_' else ""
                metadata["year"] = self._extract_year(parts[2]) if len(parts) > 2 else ""
                metadata["publishers"] = parts[3].strip() if len(parts) > 3 and parts[3].strip() != '_' else ""
                if len(parts) > 4:
                    metadata["doi"] = parts[4].strip() if parts[4].strip() not in ['_', 'Anna\'s Archive'] else ""
        else:
            # Fallback: use filename as title and try to extract year
            metadata["title"] = filename_clean.replace('_', ' ').replace('-', ' ')
            metadata["year"] = self._extract_year(filename_clean)
        
        return metadata
    
    def _extract_year(self, text: str) -> str:
        """Extract first 4-digit year from text."""
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        return year_match.group() if year_match else ""
    
    def validate_metadata(self, metadata: Dict[str, str]) -> Dict[str, str]:
        """Validate and clean extracted metadata."""
        validated = {}
        
        # Title validation
        title = metadata.get("title", "").strip()
        if len(title) < 2 or title.lower() in ["unknown", "not provided", "n/a", ""]:
            validated["title"] = ""
        elif len(title) > 200:
            validated["title"] = title[:200] + "..."
        else:
            validated["title"] = title
        
        # Authors validation
        authors = metadata.get("authors", "").strip()
        if authors.lower() in ["unknown", "not provided", "n/a", "not specified in the text", ""]:
            validated["authors"] = ""
        elif len(authors) > 150:
            # Truncate long author lists
            validated["authors"] = authors[:150] + "..."
        else:
            validated["authors"] = authors
        
        # Publishers validation
        publishers = metadata.get("publishers", "").strip()
        if publishers.lower() in ["unknown", "not provided", "n/a", "publisher name", "not specified in the text", ""]:
            validated["publishers"] = ""
        elif len(publishers) > 100:
            validated["publishers"] = publishers[:100] + "..."
        else:
            validated["publishers"] = publishers
        
        # Year validation
        year = self._extract_year(metadata.get("year", ""))
        validated["year"] = year
        
        # DOI validation
        doi = metadata.get("doi", "").strip()
        if doi.lower() in ["unknown", "not provided", "n/a", "doi number", "none", ""]:
            validated["doi"] = ""
        elif len(doi) > 50:
            validated["doi"] = doi[:50]
        else:
            validated["doi"] = doi
        
        return validated
    
    def extract_metadata_with_ollama(self, sample_text: str, max_retries: int = 2) -> Dict[str, str]:
        """Enhanced metadata extraction with retries and improved prompts."""
        
        for attempt in range(max_retries + 1):
            try:
                # Improved prompt with specific instructions
                prompt = f"""Extract metadata from this academic/technical document text and return ONLY valid JSON.

TEXT:
{sample_text[:2000]}

Return JSON with these exact keys: title, authors, publishers, year, doi

INSTRUCTIONS:
- title: The main title of the document (not chapter/section titles)
- authors: Full author names separated by commas (not "Unknown" or placeholders)
- publishers: Publishing company name (not "Publisher Name" or placeholders)  
- year: 4-digit publication year ONLY (like 2024, not ranges)
- doi: DOI, ISBN, or similar identifier (not "DOI Number" or placeholders)
- If any field is truly unknown, use empty string ""
- Do not use placeholder text like "Unknown", "Not provided", "Publisher Name"
- Return ONLY the JSON object, no other text

Example: {{"title": "Machine Learning Fundamentals", "authors": "John Smith, Jane Doe", "publishers": "MIT Press", "year": "2024", "doi": "978-0262046824"}}"""

                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "mistral:latest",
                        "prompt": prompt,
                        "options": {
                            "temperature": 0.1,
                            "max_tokens": 256,
                            "stop": ["\n\n"]
                        },
                        "stream": False
                    },
                    timeout=30
                )
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                response_text = result.get("response", "").strip()
                
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    metadata = json.loads(json_match.group())
                    
                    # Validate required keys exist
                    required_keys = ["title", "authors", "publishers", "year", "doi"]
                    if all(key in metadata for key in required_keys):
                        validated = self.validate_metadata(metadata)
                        print(f"    ✅ Ollama extraction successful (attempt {attempt + 1})")
                        return validated
                    else:
                        print(f"    ❌ Missing required keys in attempt {attempt + 1}: {metadata}")
                        if attempt == max_retries:
                            return self._empty_metadata()
                        continue
                else:
                    print(f"    ❌ No JSON found in response (attempt {attempt + 1}): {response_text[:100]}")
                    if attempt == max_retries:
                        return self._empty_metadata()
                    continue
                    
            except json.JSONDecodeError as e:
                print(f"    ❌ JSON decode error (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    return self._empty_metadata()
                continue
            except Exception as e:
                print(f"    ❌ Ollama request error (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    return self._empty_metadata()
                continue
        
        return self._empty_metadata()
    
    def _empty_metadata(self) -> Dict[str, str]:
        """Return empty metadata structure."""
        return {
            "title": "",
            "authors": "",
            "publishers": "",
            "year": "",
            "doi": ""
        }
    
    def extract_metadata_combined(self, pdf_path: Path) -> Dict[str, str]:
        """Combined extraction: Ollama + filename fallback + validation."""
        print(f"📄 Extracting metadata from: {pdf_path.name}")
        
        # Step 1: Extract text sample
        sample_text = self.extract_text_sample(pdf_path)
        if not sample_text:
            print("    ❌ No text extracted, using filename only")
            return self.validate_metadata(self.extract_metadata_from_filename(pdf_path.name))
        
        print(f"    📝 Text sample: {len(sample_text)} chars")
        
        # Step 2: Try Ollama extraction
        ollama_metadata = self.extract_metadata_with_ollama(sample_text)
        
        # Step 3: Fallback to filename if Ollama failed
        filename_metadata = self.extract_metadata_from_filename(pdf_path.name)
        
        # Step 4: Combine results (Ollama preferred, filename as fallback)
        combined = {}
        for key in ["title", "authors", "publishers", "year", "doi"]:
            ollama_value = ollama_metadata.get(key, "").strip()
            filename_value = filename_metadata.get(key, "").strip()
            
            # Use Ollama value if meaningful, otherwise filename value
            if ollama_value and len(ollama_value) > 1:
                combined[key] = ollama_value
            elif filename_value and len(filename_value) > 1:
                combined[key] = filename_value
                print(f"    🔄 Used filename for {key}: {filename_value}")
            else:
                combined[key] = ""
        
        # Final validation
        final_metadata = self.validate_metadata(combined)
        
        # Quality check
        non_empty_fields = sum(1 for v in final_metadata.values() if v.strip())
        print(f"    📊 Extracted {non_empty_fields}/5 fields successfully")
        
        return final_metadata


def test_metadata_extraction():
    """Test the improved metadata extraction on sample files."""
    
    # Check if Ollama is available
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        ollama_available = response.status_code == 200
    except:
        ollama_available = False
    
    if not ollama_available:
        print("❌ Ollama not available at localhost:11434")
        print("   Start Ollama server to test LLM extraction")
        return
    
    extractor = ImprovedMetadataExtractor()
    test_dir = Path("test_metadata")
    
    if not test_dir.exists():
        print("❌ test_metadata directory not found")
        return
    
    pdf_files = list(test_dir.glob("*.pdf"))[:5]  # Test first 5 files
    
    if not pdf_files:
        print("❌ No PDF files found in test_metadata/")
        return
    
    print(f"🧪 Testing metadata extraction on {len(pdf_files)} files...")
    print()
    
    results = []
    for pdf_path in pdf_files:
        metadata = extractor.extract_metadata_combined(pdf_path)
        results.append({
            "filename": pdf_path.name,
            "metadata": metadata
        })
        
        print(f"    📚 Title: {metadata['title'][:60]}{'...' if len(metadata['title']) > 60 else ''}")
        print(f"    👤 Authors: {metadata['authors'][:50]}{'...' if len(metadata['authors']) > 50 else ''}")
        print(f"    🏢 Publishers: {metadata['publishers'][:40]}{'...' if len(metadata['publishers']) > 40 else ''}")
        print(f"    📅 Year: {metadata['year']}")
        print(f"    🔗 DOI: {metadata['doi'][:30]}{'...' if len(metadata['doi']) > 30 else ''}")
        print()
    
    # Summary
    total_fields = len(results) * 5
    extracted_fields = sum(
        sum(1 for v in result["metadata"].values() if v.strip())
        for result in results
    )
    
    print(f"📊 SUMMARY:")
    print(f"   Files tested: {len(results)}")
    print(f"   Fields extracted: {extracted_fields}/{total_fields} ({extracted_fields/total_fields*100:.1f}%)")
    print(f"   Average fields per file: {extracted_fields/len(results):.1f}/5")


if __name__ == "__main__":
    test_metadata_extraction()