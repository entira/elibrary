#!/usr/bin/env python3
"""
MetadataExtractor Module

Handles intelligent metadata extraction from academic documents using LLM
with optimizations for gemma3:4b-it-qat model, validation, and fallback mechanisms.
"""

import json
import re
import warnings
from pathlib import Path
from typing import Dict, Optional
import requests

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


class MetadataExtractor:
    """Enhanced metadata extraction optimized for gemma3:4b-it-qat model."""
    
    def __init__(self, 
                 model: str = "gemma3:4b-it-qat",
                 base_url: str = "http://localhost:11434",
                 max_retries: int = 2):
        """Initialize MetadataExtractor.
        
        Args:
            model: Ollama model to use for extraction
            base_url: Ollama server URL
            max_retries: Maximum retry attempts
        """
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
    
    def extract_metadata(self, text: str, filename: str = "") -> Dict[str, str]:
        """Extract metadata from document text with LLM and filename fallback.
        
        Args:
            text: Document text (preferably first page)
            filename: PDF filename for fallback extraction
            
        Returns:
            Dictionary with extracted metadata
        """
        return self.extract_metadata_with_ollama(text, filename, self.max_retries)
    
    def extract_metadata_with_ollama(self, sample_text: str, filename: str = "", max_retries: int = 2) -> Dict[str, str]:
        """Enhanced metadata extraction optimized for gemma3:4b-it-qat model.
        
        Args:
            sample_text: Text to extract metadata from
            filename: Filename for fallback extraction
            max_retries: Maximum retry attempts
            
        Returns:
            Extracted and validated metadata dictionary
        """
        
        for attempt in range(max_retries + 1):
            try:
                # Build adaptive prompt based on attempt
                prompt = self.build_extraction_prompt(sample_text, attempt + 1)
                
                # Improved inference configuration for gemma3:4b-it-qat
                ollama_request = {
                    "model": self.model,
                    "prompt": prompt,
                    "options": {
                        "temperature": 0,        # Deterministic output
                        "top_p": 0.9,           # Nucleus sampling
                        "max_tokens": 512       # Increased token limit
                        # Removed stop token to prevent truncation
                    },
                    "stream": False
                }
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=ollama_request,
                    timeout=45  # Increased timeout for larger models
                )
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                response_text = result.get("response", "").strip()
                
                # Use robust JSON parsing
                metadata = self.parse_json_response(response_text)
                
                if metadata:
                    # Validate required keys exist
                    required_keys = ["title", "authors", "publishers", "year", "doi"]
                    if all(key in metadata for key in required_keys):
                        # Normalize each field
                        normalized_metadata = {}
                        for key in required_keys:
                            raw_value = str(metadata.get(key, ""))
                            normalized_metadata[key] = self.normalize_metadata_field(key, raw_value)
                        
                        # Validate after normalization
                        validated_metadata = self.validate_metadata(normalized_metadata)
                        
                        # Combine with filename fallback if provided
                        if filename:
                            filename_metadata = self.extract_metadata_from_filename(filename)
                            final_metadata = {}
                            for key in required_keys:
                                ollama_value = validated_metadata.get(key, "").strip()
                                filename_value = filename_metadata.get(key, "").strip()
                                
                                # Use Ollama value if meaningful, otherwise filename value
                                if ollama_value and len(ollama_value) > 1:
                                    final_metadata[key] = ollama_value
                                elif filename_value and len(filename_value) > 1:
                                    final_metadata[key] = filename_value
                                    print(f"     🔄 Used filename for {key}")
                                else:
                                    final_metadata[key] = ""
                            
                            print(f"     ✅ Enhanced extraction successful (attempt {attempt + 1})")
                            return final_metadata
                        else:
                            print(f"     ✅ Ollama extraction successful (attempt {attempt + 1})")
                            return validated_metadata
                    else:
                        print(f"     ❌ Missing required keys in attempt {attempt + 1}: {list(metadata.keys())}")
                        if attempt == max_retries:
                            break
                        continue
                else:
                    print(f"     ❌ No valid JSON found in response (attempt {attempt + 1})")
                    print(f"     🔍 Response preview: {response_text[:200]}...")
                    if attempt == max_retries:
                        break
                    continue
                    
            except json.JSONDecodeError as e:
                print(f"     ❌ JSON decode error (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    break
                continue
            except Exception as e:
                print(f"     ❌ Ollama request error (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    break
                continue
        
        # If all attempts failed, use filename fallback
        if filename:
            print(f"     🔄 All Ollama attempts failed, using filename fallback")
            return self.validate_metadata(self.extract_metadata_from_filename(filename))
        else:
            return self._empty_metadata()
    
    def build_extraction_prompt(self, document_text: str, attempt: int = 1) -> str:
        """Build optimized prompt for gemma3:4b-it-qat model.
        
        Args:
            document_text: Text to extract metadata from
            attempt: Attempt number (affects prompt strategy)
            
        Returns:
            Optimized prompt string
        """
        if attempt == 1:
            # First attempt: Clean, focused prompt
            return f"""Extract the following metadata from the academic text below. Respond with valid JSON only.

TEXT:
{document_text}

OUTPUT JSON with this exact structure:
{{
  "title": "...",
  "authors": "...",
  "publishers": "...",
  "year": "...",
  "doi": "..."
}}

INSTRUCTIONS:
- title: The main title of the document
- authors: Full author names, comma-separated
- publishers: Full publisher name
- year: 4-digit year only (e.g. "2023")
- doi: DOI, ISBN, or similar (e.g. "10.1145/1234567")

If any field is truly unknown, return an empty string.
Do NOT include explanations or examples. Return the JSON object only."""
        
        else:
            # Retry attempts: Add few-shot example for better guidance
            return f"""You are a strict JSON parser for academic metadata extraction. Return valid JSON and nothing else.

Extract metadata from this text:

TEXT:
{document_text}

Example output format:
{{"title": "Deep Learning Methods", "authors": "Alice Johnson, Bob Smith", "publishers": "Academic Press", "year": "2023", "doi": "10.1016/example"}}

Your JSON output:"""
    
    def parse_json_response(self, response_text: str) -> Dict[str, str]:
        """Robust JSON parsing with precise bracket matching.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed JSON dictionary or empty dict if parsing fails
        """
        try:
            # Find first and last braces for precise extraction
            first_brace = response_text.find('{')
            last_brace = response_text.rfind('}')
            
            if first_brace == -1 or last_brace == -1 or first_brace >= last_brace:
                return {}
            
            json_str = response_text[first_brace:last_brace + 1]
            return json.loads(json_str)
            
        except (json.JSONDecodeError, ValueError):
            # Fallback: try to find JSON using regex (less reliable)
            try:
                json_match = re.search(r'\\{[^}]*\\}', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except (json.JSONDecodeError, ValueError):
                pass
        
        return {}
    
    def normalize_metadata_field(self, field_name: str, value: str) -> str:
        """Normalize specific metadata fields post-extraction.
        
        Args:
            field_name: Name of the metadata field
            value: Raw value to normalize
            
        Returns:
            Normalized field value
        """
        if not value or not isinstance(value, str):
            return ""
        
        value = value.strip()
        
        if field_name == "year":
            # Extract 4-digit year
            year_match = re.search(r'\\b(19|20)\\d{2}\\b', value)
            return year_match.group() if year_match else ""
        
        elif field_name == "doi":
            # Clean DOI - remove prefixes and normalize
            cleaned = value.lower().replace("doi:", "").replace("isbn:", "").strip()
            # Remove common placeholder patterns
            if cleaned in ["unknown", "not provided", "n/a", "doi number", "none"]:
                return ""
            return cleaned if len(cleaned) > 3 else ""
        
        elif field_name in ["title", "authors", "publishers"]:
            # Remove common placeholder patterns
            if value.lower() in ["unknown", "not provided", "n/a", "publisher name", "author name", "title"]:
                return ""
            return value
        
        return value
    
    def extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """Fallback: extract metadata from filename patterns.
        
        Args:
            filename: PDF filename to parse
            
        Returns:
            Metadata dictionary extracted from filename
        """
        metadata = {
            "title": "",
            "authors": "",
            "publishers": "",
            "year": "",
            "doi": ""
        }
        
        if not filename:
            return metadata
        
        # Remove file extension
        name = Path(filename).stem
        
        # Try to parse structured filename (common patterns)
        # Pattern 1: "Title -- Author -- Year -- Publisher -- ISBN"
        parts = [part.strip() for part in name.split(' -- ')]
        if len(parts) >= 4:
            metadata["title"] = parts[0]
            metadata["authors"] = parts[1] 
            metadata["year"] = self._extract_year(parts[2])
            metadata["publishers"] = parts[3]
            if len(parts) >= 5:
                metadata["doi"] = parts[4]
            return metadata
        
        # Pattern 2: "Author_Title_Year.pdf"
        underscore_parts = name.split('_')
        if len(underscore_parts) >= 3:
            metadata["authors"] = underscore_parts[0].replace('_', ' ')
            metadata["title"] = underscore_parts[1].replace('_', ' ')
            metadata["year"] = self._extract_year(underscore_parts[2])
            return metadata
        
        # Pattern 3: Extract what we can from any pattern
        # Look for years anywhere in filename
        year_match = re.search(r'\b(19|20)\d{2}\b', name)
        if year_match:
            metadata["year"] = year_match.group()
        
        # Use filename as title if nothing else found
        if not metadata["title"] and len(name) > 5:
            # Clean up the filename for use as title
            title = re.sub(r'[_-]', ' ', name)
            title = re.sub(r'\s+', ' ', title)
            metadata["title"] = title.strip()
        
        return metadata
    
    def validate_metadata(self, metadata: Dict[str, str]) -> Dict[str, str]:
        """Validate and clean extracted metadata.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Validated and cleaned metadata
        """
        validated = {}
        
        # Title validation
        title = metadata.get("title", "").strip()
        if title.lower() in ["unknown", "not provided", "n/a", "title", ""]:
            validated["title"] = ""
        elif len(title) > 200:
            validated["title"] = title[:200]
        else:
            validated["title"] = title
        
        # Authors validation
        authors = metadata.get("authors", "").strip()
        if authors.lower() in ["unknown", "not provided", "n/a", "author", "authors", ""]:
            validated["authors"] = ""
        elif len(authors) > 150:
            validated["authors"] = authors[:150]
        else:
            validated["authors"] = authors
        
        # Publishers validation
        publishers = metadata.get("publishers", "").strip()
        if publishers.lower() in ["unknown", "not provided", "n/a", "publisher", "publishers", "publisher name", ""]:
            validated["publishers"] = ""
        elif len(publishers) > 100:
            validated["publishers"] = publishers[:100]
        else:
            validated["publishers"] = publishers
        
        # Year validation
        year = metadata.get("year", "").strip()
        year_match = re.search(r'\b(19|20)\d{2}\b', year)
        if year_match:
            validated["year"] = year_match.group()
        else:
            validated["year"] = ""
        
        # DOI validation
        doi = metadata.get("doi", "").strip()
        if doi.lower() in ["unknown", "not provided", "n/a", "doi number", "none", ""]:
            validated["doi"] = ""
        elif len(doi) > 50:
            validated["doi"] = doi[:50]
        else:
            validated["doi"] = doi
        
        return validated
    
    def _extract_year(self, year_text: str) -> str:
        """Extract first 4-digit year from text.
        
        Args:
            year_text: Text potentially containing year
            
        Returns:
            4-digit year string or empty string
        """
        year_match = re.search(r'\b(19|20)\d{2}\b', year_text)
        return year_match.group() if year_match else ""
    
    def _empty_metadata(self) -> Dict[str, str]:
        """Return empty metadata structure.
        
        Returns:
            Empty metadata dictionary with all required fields
        """
        return {
            "title": "",
            "authors": "",
            "publishers": "",
            "year": "",
            "doi": ""
        }


# Utility functions for standalone usage
def extract_metadata_from_text(text: str, filename: str = "") -> Dict[str, str]:
    """Convenience function to extract metadata from text.
    
    Args:
        text: Document text
        filename: Optional filename for fallback
        
    Returns:
        Extracted metadata dictionary
    """
    extractor = MetadataExtractor()
    return extractor.extract_metadata(text, filename)


def extract_metadata_from_file(file_path: str) -> Dict[str, str]:
    """Convenience function to extract metadata from text file.
    
    Args:
        file_path: Path to text file
        
    Returns:
        Extracted metadata dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    filename = Path(file_path).name
    return extract_metadata_from_text(text, filename)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python metadata_extractor.py <text_file>")
        print("       python metadata_extractor.py - (read from stdin)")
        sys.exit(1)
    
    # Read input
    if sys.argv[1] == "-":
        text = sys.stdin.read()
        filename = ""
    else:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            text = f.read()
        filename = Path(sys.argv[1]).name
    
    print(f"Extracting metadata from text ({len(text)} characters)...")
    if filename:
        print(f"Filename: {filename}")
    
    extractor = MetadataExtractor()
    metadata = extractor.extract_metadata(text, filename)
    
    print("\\n📊 Extracted Metadata:")
    for key, value in metadata.items():
        print(f"   {key}: {value}")
    
    # Calculate extraction quality
    non_empty_fields = sum(1 for v in metadata.values() if v.strip())
    quality = (non_empty_fields / len(metadata)) * 100
    print(f"\\n📈 Extraction quality: {quality:.1f}% ({non_empty_fields}/{len(metadata)} fields)")