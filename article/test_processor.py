#!/usr/bin/env python3
"""
Model Comparison Test Processor
Tests metadata extraction with different Gemma models without video generation.

Models to test:
- gemma3:4b
- gemma3:12b  
- gemma3:4b-it-qat
- gemma3:12b-it-qat

Usage:
    python3 test_processor.py
"""

# Suppress ALL warnings and output before any imports
import warnings
import sys
import os
from io import StringIO
import contextlib

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Capture and suppress stdout during problematic imports
def suppress_stdout():
    return contextlib.redirect_stdout(StringIO())

def suppress_stderr():  
    return contextlib.redirect_stderr(StringIO())

import json
import re
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pymupdf as fitz
from tqdm import tqdm
import time

# Models to test in sequence
MODELS_TO_TEST = [
    "mistral:latest"
]

class ModelTester:
    """Test metadata extraction with different models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.results = []
        self.stats = {
            "total_files": 0,
            "successful_extractions": 0,
            "json_decode_errors": 0,
            "missing_keys_errors": 0,
            "no_json_errors": 0,
            "other_errors": 0,
            "fallback_used": 0,
            "total_processing_time": 0.0,
            "average_time_per_file": 0.0
        }
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Validate and clean metadata fields."""
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
        year = str(metadata.get("year", "")).strip()
        if re.match(r'^\d{4}$', year):
            validated["year"] = year
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

    def extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """Extract metadata from PDF filename as fallback."""
        clean_name = Path(filename).stem
        clean_name = clean_name.replace("_", " ").replace("-", " ")
        clean_name = re.sub(r'\d{10,}', '', clean_name)  # Remove long numbers
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
        
        # Try to extract year from filename
        year_match = re.search(r'(19|20)\d{2}', clean_name)
        year = year_match.group() if year_match else ""
        
        # Try to extract DOI-like patterns
        doi_match = re.search(r'(978[\d\-]+|\d{4}-\d{4})', filename)
        doi = doi_match.group() if doi_match else ""
        
        return {
            "title": clean_name[:100] if clean_name else "",
            "authors": "",
            "publishers": "",
            "year": year,
            "doi": doi
        }

    def extract_metadata_with_ollama(self, sample_text: str, filename: str = "", max_retries: int = 2) -> Dict[str, str]:
        """Extract metadata using specific model with retries, validation and filename fallback."""
        
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
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model_name,
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
                        ollama_metadata = self.validate_metadata(metadata)
                        
                        # Combine with filename fallback if provided
                        if filename:
                            filename_metadata = self.extract_metadata_from_filename(filename)
                            final_metadata = {}
                            for key in required_keys:
                                ollama_value = ollama_metadata.get(key, "").strip()
                                filename_value = filename_metadata.get(key, "").strip()
                                
                                # Use Ollama value if meaningful, otherwise filename value
                                if ollama_value and len(ollama_value) > 1:
                                    final_metadata[key] = ollama_value
                                elif filename_value and len(filename_value) > 1:
                                    final_metadata[key] = filename_value
                                else:
                                    final_metadata[key] = ""
                            
                            print(f"     ✅ Enhanced extraction successful (attempt {attempt + 1})")
                            self.stats["successful_extractions"] += 1
                            return final_metadata
                        else:
                            print(f"     ✅ Ollama extraction successful (attempt {attempt + 1})")
                            self.stats["successful_extractions"] += 1
                            return ollama_metadata
                    else:
                        print(f"     ❌ Missing required keys in attempt {attempt + 1}")
                        self.stats["missing_keys_errors"] += 1
                        if attempt == max_retries:
                            break
                        continue
                else:
                    print(f"     ❌ No JSON found in response (attempt {attempt + 1})")
                    self.stats["no_json_errors"] += 1
                    if attempt == max_retries:
                        break
                    continue
                    
            except json.JSONDecodeError as e:
                print(f"     ❌ JSON decode error (attempt {attempt + 1}): {e}")
                self.stats["json_decode_errors"] += 1
                if attempt == max_retries:
                    break
                continue
                
            except Exception as e:
                print(f"     ❌ Error in attempt {attempt + 1}: {e}")
                self.stats["other_errors"] += 1
                if attempt == max_retries:
                    break
                continue
        
        # All attempts failed, use filename fallback
        print(f"     🔄 All Ollama attempts failed, using filename fallback")
        self.stats["fallback_used"] += 1
        
        if filename:
            fallback_metadata = self.extract_metadata_from_filename(filename)
            return fallback_metadata
        else:
            return {
                "title": "",
                "authors": "",
                "publishers": "",
                "year": "",
                "doi": ""
            }

    def test_pdf_file(self, pdf_path: Path, library_id: str = "") -> Dict[str, Any]:
        """Test metadata extraction on a single PDF file."""
        print(f"   🔍 Testing: {pdf_path.name}")
        self.stats["total_files"] += 1
        
        # Start timing for this file
        file_start_time = time.time()
        
        try:
            # Extract text from first few pages for sampling
            doc = fitz.open(pdf_path)
            sample_text = ""
            
            # Get text from first 3 pages or all pages if fewer
            max_pages = min(3, len(doc))
            for page_num in range(max_pages):
                page = doc[page_num]
                page_text = page.get_text()
                sample_text += page_text + "\n"
            
            doc.close()
            
            # Limit sample text to prevent huge prompts
            sample_text = sample_text[:3000]
            
            if not sample_text.strip():
                print(f"     ⚠️ No text extracted from {pdf_path.name}")
                file_total_time = time.time() - file_start_time
                self.stats["total_processing_time"] += file_total_time
                return {
                    "filename": pdf_path.name,
                    "status": "no_text",
                    "metadata": self.extract_metadata_from_filename(pdf_path.name),
                    "error": "No text extracted",
                    "library_id": library_id,
                    "processing_time": file_total_time,
                    "extraction_time": 0.0
                }
            
            # Extract metadata
            print(f"     🤖 Extracting metadata with {self.model_name}...")
            extraction_start = time.time()
            metadata = self.extract_metadata_with_ollama(sample_text, pdf_path.name)
            extraction_time = time.time() - extraction_start
            file_total_time = time.time() - file_start_time
            
            # Update timing stats
            self.stats["total_processing_time"] += file_total_time
            
            result = {
                "filename": pdf_path.name,
                "status": "success",
                "metadata": metadata,
                "sample_text_length": len(sample_text),
                "library_id": library_id,
                "processing_time": file_total_time,
                "extraction_time": extraction_time
            }
            
            # Log result summary
            print(f"     📚 Title: {metadata['title'][:60]}{'...' if len(metadata['title']) > 60 else ''}")
            print(f"     👤 Authors: {metadata['authors'][:50]}{'...' if len(metadata['authors']) > 50 else ''}")
            print(f"     📅 Year: {metadata['year']}")
            print(f"     ⏱️ Time: {file_total_time:.2f}s (extraction: {extraction_time:.2f}s)")
            
            return result
            
        except Exception as e:
            print(f"     ❌ Error processing {pdf_path.name}: {e}")
            self.stats["other_errors"] += 1
            file_total_time = time.time() - file_start_time
            self.stats["total_processing_time"] += file_total_time
            return {
                "filename": pdf_path.name,
                "status": "error",
                "metadata": self.extract_metadata_from_filename(pdf_path.name),
                "error": str(e),
                "library_id": library_id,
                "processing_time": file_total_time,
                "extraction_time": 0.0
            }

    def test_library(self, library_path: str, max_files: int = None) -> None:
        """Test metadata extraction on a library of PDFs."""
        pdf_dir = Path(library_path) / "pdf"
        
        if not pdf_dir.exists():
            print(f"❌ PDF directory not found: {pdf_dir}")
            return
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in {pdf_dir}")
            return
        
        # Use all files unless max_files is specified
        if max_files is not None:
            test_files = pdf_files[:max_files]
        else:
            test_files = pdf_files
        
        print(f"🔬 Testing {self.model_name} on {len(test_files)} files from {library_path}")
        print(f"📂 PDF Directory: {pdf_dir}")
        print()
        
        start_time = time.time()
        
        # Extract library ID from path (e.g., "library/1" -> "1")
        library_id = Path(library_path).name
        
        for i, pdf_path in enumerate(test_files, 1):
            print(f"   [{i}/{len(test_files)}] ", end="")
            result = self.test_pdf_file(pdf_path, library_id)
            self.results.append(result)
            print()  # Add spacing between files
        
        processing_time = time.time() - start_time
        
        # Calculate success rate and timing stats
        successful = self.stats["successful_extractions"]
        total = self.stats["total_files"]
        success_rate = (successful / total * 100) if total > 0 else 0
        
        # Update average time per file
        if total > 0:
            self.stats["average_time_per_file"] = self.stats["total_processing_time"] / total
        
        print(f"⏱️ Processing time: {processing_time:.2f}s")
        print(f"📊 Success rate: {successful}/{total} ({success_rate:.1f}%)")
        print(f"⚡ Average per file: {self.stats['average_time_per_file']:.2f}s")
        print(f"📈 Stats: JSON errors: {self.stats['json_decode_errors']}, Missing keys: {self.stats['missing_keys_errors']}, No JSON: {self.stats['no_json_errors']}, Fallback: {self.stats['fallback_used']}")
        print()

    def save_results(self, output_file: str) -> None:
        """Save test results to JSON file."""
        # Separate results by library using library_id
        library1_results = [r for r in self.results if r.get("library_id") == "1"]
        library2_results = [r for r in self.results if r.get("library_id") == "2"]
        
        # Calculate performance metrics
        total_files = len(self.results)
        total_time = self.stats.get("total_processing_time", 0)
        avg_time = total_time / total_files if total_files > 0 else 0
        
        # Calculate timing stats per library
        lib1_times = [r.get("processing_time", 0) for r in library1_results if r.get("processing_time")]
        lib2_times = [r.get("processing_time", 0) for r in library2_results if r.get("processing_time")]
        
        lib1_avg_time = sum(lib1_times) / len(lib1_times) if lib1_times else 0
        lib2_avg_time = sum(lib2_times) / len(lib2_times) if lib2_times else 0
        
        output_data = {
            "model": self.model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_stats": self.stats,
            "performance_metrics": {
                "total_files": total_files,
                "total_processing_time": total_time,
                "average_time_per_file": avg_time,
                "library1_avg_time": lib1_avg_time,
                "library2_avg_time": lib2_avg_time,
                "files_per_minute": (total_files / (total_time / 60)) if total_time > 0 else 0
            },
            "library_breakdown": {
                "library1": {
                    "count": len(library1_results),
                    "avg_processing_time": lib1_avg_time,
                    "results": library1_results
                },
                "library2": {
                    "count": len(library2_results), 
                    "avg_processing_time": lib2_avg_time,
                    "results": library2_results
                }
            },
            "all_results": self.results,
            "notes": "Results from both Library 1 (books) and Library 2 (ebooks/papers) with timing analysis"
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_path}")


def check_ollama_connection() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def check_model_availability(model_name: str) -> bool:
    """Check if a specific model is available in Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            available_models = [model["name"] for model in data.get("models", [])]
            return model_name in available_models
        return False
    except:
        return False


def main():
    """Main entry point for model testing."""
    print("🧪 Model Comparison Test Processor")
    print("=" * 50)
    
    # Check Ollama connection
    if not check_ollama_connection():
        print("❌ Ollama is not running at localhost:11434")
        print("   Please start Ollama server and try again.")
        sys.exit(1)
    
    print("✅ Ollama connection confirmed")
    
    # Check which models are available
    available_models = []
    unavailable_models = []
    
    for model in MODELS_TO_TEST:
        if check_model_availability(model):
            available_models.append(model)
            print(f"✅ Model available: {model}")
        else:
            unavailable_models.append(model)
            print(f"❌ Model not available: {model}")
    
    if unavailable_models:
        print(f"\n⚠️ Missing models: {', '.join(unavailable_models)}")
        print("   Run: ollama pull <model_name> to download missing models")
        
        if not available_models:
            print("❌ No models available for testing")
            sys.exit(1)
        else:
            print(f"✅ Proceeding with {len(available_models)} available models")
    
    print()
    
    # Test each available model
    for i, model_name in enumerate(available_models, 1):
        print(f"🚀 Testing model {i}/{len(available_models)}: {model_name}")
        print("-" * 50)
        
        tester = ModelTester(model_name)
        
        # Test on both libraries for comprehensive results
        print(f"📚 Testing Library 1 (Books)...")
        tester.test_library("library/1")
        
        print(f"📚 Testing Library 2 (Ebooks/Papers)...")  
        tester.test_library("library/2")
        
        # Generate output filename
        safe_model_name = model_name.replace(":", "_").replace("/", "_")
        output_file = f"test_results_{safe_model_name}.json"
        
        # Save results
        tester.save_results(output_file)
        
        print("=" * 50)
        print()
    
    print("🎉 All model testing completed!")
    print(f"📁 Results saved to test_results_*.json files")
    
    # Create summary
    print("\n📊 SUMMARY:")
    for model in available_models:
        safe_model_name = model.replace(":", "_").replace("/", "_")
        result_file = f"test_results_{safe_model_name}.json"
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                stats = data["stats"]
                total = stats["total_files"]
                successful = stats["successful_extractions"]
                success_rate = (successful / total * 100) if total > 0 else 0
                avg_time = data.get("performance_metrics", {}).get("average_time_per_file", 0)
                files_per_min = data.get("performance_metrics", {}).get("files_per_minute", 0)
                
                print(f"   {model}: {successful}/{total} ({success_rate:.1f}%) success rate, {avg_time:.2f}s/file, {files_per_min:.1f} files/min")
        except:
            print(f"   {model}: Failed to load results")


if __name__ == "__main__":
    main()