#!/usr/bin/env python3
"""
PDF Library Processor with Enhanced Skip Optimization
Processes PDF books with detailed page tracking and intelligent skip mechanism.

Features:
- Smart skip mechanism - avoids reprocessing already processed PDFs
- Enhanced page metadata with cross-page context preservation  
- Token-based sliding window chunking for optimal RAG performance
- Clean warning suppression for faster testing
- No pickle errors from simplified implementation

Usage:
    python3 pdf_library_processor.py                       # Skip already processed PDFs  
    python3 pdf_library_processor.py --force-reprocess     # Force reprocess all PDFs
    python3 pdf_library_processor.py --max-workers 8       # Use 8 workers for processing
    
    For completely clean output without any warnings:
    python3 pdf_processor_quiet.py --max-workers 8         # Uses quiet wrapper
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

# Import memvid only when needed to avoid dependency issues

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import tiktoken


class OllamaEmbedder:
    """Embedding class using Ollama's nomic-embed-text model."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "nomic-embed-text"
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = []
        
        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": text,
                        "embedding": True
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                
                if "embedding" in result:
                    embeddings.append(result["embedding"])
                else:
                    print(f"Warning: No embedding in response for text: {text[:50]}...")
                    embeddings.append([0.0] * 768)  # Default embedding size
                    
            except Exception as e:
                print(f"Error getting embedding for text: {e}")
                embeddings.append([0.0] * 768)  # Default embedding size
        
        return embeddings


def generate_single_qr_global(args):
    """Global function for QR generation to avoid pickle issues."""
    # Aggressively suppress ALL warnings in worker processes
    import warnings
    warnings.filterwarnings("ignore")
    import os
    import sys
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Redirect stderr to suppress any remaining warnings
    import contextlib
    with contextlib.redirect_stderr(open(os.devnull, 'w')):
        frame_num, chunk_text, frames_dir_str, config = args
    try:
        import qrcode
        import json
        import os
        
        chunk_data = {"id": frame_num, "text": chunk_text, "frame": frame_num}
        
        # Use memvid's QR config with auto-truncation for long chunks
        qr_config = config.get('qr', {})
        
        # Try with progressively shorter text if QR version exceeds 40
        original_text = chunk_text
        for max_chars in [len(chunk_text), 2800, 2400, 2000, 1600, 1200]:
            try:
                if max_chars < len(chunk_text):
                    chunk_data["text"] = chunk_text[:max_chars] + "..."
                else:
                    chunk_data["text"] = chunk_text
                    
                qr = qrcode.QRCode(
                    version=1,  # Let it auto-scale but with limit
                    error_correction=getattr(qrcode.constants, f"ERROR_CORRECT_{qr_config.get('error_correction', 'M')}"),
                    box_size=qr_config.get('box_size', 5),
                    border=qr_config.get('border', 3)
                )
                qr.add_data(json.dumps(chunk_data))
                qr.make(fit=True)
                
                # Check if version is acceptable
                if qr.version <= 40:
                    break
            except Exception:
                continue
        else:
            # Fallback: use minimal data
            chunk_data["text"] = f"[Chunk {frame_num} - Text too long for QR]"
            qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_M)
            qr.add_data(json.dumps(chunk_data))
            qr.make(fit=True)
        
        qr_image = qr.make_image(
            fill_color=qr_config.get('fill_color', 'black'),
            back_color=qr_config.get('back_color', 'white')
        )
        
        frame_path = os.path.join(frames_dir_str, f"frame_{frame_num:06d}.png")
        qr_image.save(frame_path)
        
        return frame_num
    except Exception as e:
        print(f"Error generating QR frame {frame_num}: {e}")
        return None


def monkey_patch_parallel_qr_generation(encoder, n_workers: int):
    """Monkey patch MemvidEncoder to use parallel QR generation."""
    import types
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor
    
    def _generate_qr_frames_parallel(self, temp_dir: Path, show_progress: bool = True) -> Path:
        """Generate QR frames in parallel using ProcessPoolExecutor."""
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()
        
        # Prepare data for parallel processing
        chunk_tasks = [(i, chunk, str(frames_dir), self.config) 
                      for i, chunk in enumerate(self.chunks)]
        
        print(f"🚀 Generating {len(chunk_tasks)} QR frames using {n_workers} workers...")
        
        # Generate QR frames in parallel using global function
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            if show_progress:
                # Show progress with suppressed worker stderr
                import contextlib
                original_stderr = sys.stderr
                try:
                    # Temporarily suppress stderr only for worker warnings, keep tqdm visible
                    with open(os.devnull, 'w') as devnull:
                        sys.stderr = devnull
                        results = list(tqdm(executor.map(generate_single_qr_global, chunk_tasks), 
                                           total=len(chunk_tasks), desc="Generating QR frames", file=sys.stdout))
                finally:
                    sys.stderr = original_stderr
            else:
                list(executor.map(generate_single_qr_global, chunk_tasks))
        
        return frames_dir
    
    # Replace the original method with our parallel version
    encoder._generate_qr_frames = types.MethodType(_generate_qr_frames_parallel, encoder)
    encoder.n_workers = n_workers
    
    return encoder


def get_processed_pdfs_from_index(index_path: Path) -> set:
    """Get list of already processed PDFs from existing index."""
    try:
        if not index_path.exists():
            return set()
        
        with open(index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_pdfs = set()
        if 'enhanced_stats' in data and 'files' in data['enhanced_stats']:
            for filename in data['enhanced_stats']['files'].keys():
                processed_pdfs.add(filename)
        
        return processed_pdfs
    except Exception as e:
        print(f"Warning: Could not read existing index: {e}")
        return set()


class EnhancedChunk:
    """Enhanced chunk with detailed page metadata."""
    
    def __init__(self, text: str, start_page: int, end_page: int, 
                 chunk_index: int, total_chunks_on_pages: int):
        self.text = text.strip()
        self.start_page = start_page
        self.end_page = end_page
        self.chunk_index = chunk_index
        self.total_chunks_on_pages = total_chunks_on_pages
        self.word_count = len(text.split())
        
    def get_page_reference(self) -> str:
        """Get human-readable page reference."""
        if self.start_page == self.end_page:
            return str(self.start_page)
        else:
            return f"{self.start_page}-{self.end_page}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metadata."""
        return {
            "start_page": self.start_page,
            "end_page": self.end_page,
            "page_reference": self.get_page_reference(),
            "chunk_index": self.chunk_index,
            "total_chunks_on_pages": self.total_chunks_on_pages,
            "word_count": self.word_count
        }


class PDFLibraryProcessorV2:
    """Enhanced PDF processor with detailed page tracking and multi-library support."""
    
    def __init__(self, library_root: str = "./library", n_workers: int = None, force_reprocess: bool = False):
        self.library_root = Path(library_root)
        self.library_root.mkdir(exist_ok=True)
        self.force_reprocess = force_reprocess
        
        # Set worker count
        self.encoder_workers = n_workers if n_workers else cpu_count()
        
        # Initialize embedder
        self.embedder = OllamaEmbedder()
        
        # Initialize tokenizer for token-based chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 compatible encoding
        
        # Chunk configuration - token-based sliding window
        self.chunk_size_tokens = 500    # Target tokens per chunk (optimal for RAG)
        self.overlap_percentage = 0.15  # 15% overlap between chunks
        self.overlap_tokens = int(self.chunk_size_tokens * self.overlap_percentage)  # 75 tokens
    
    def discover_libraries(self) -> List[Dict[str, Any]]:
        """Discover all library instances with PDF content."""
        libraries = []
        
        if not self.library_root.exists():
            return libraries
        
        # Search for numbered library directories
        for item in self.library_root.iterdir():
            if item.is_dir() and item.name.isdigit():
                pdf_dir = item / "pdf"
                data_dir = item / "data"
                
                if pdf_dir.exists():
                    # Check for PDF files
                    pdf_files = list(pdf_dir.glob("*.pdf"))
                    if pdf_files:
                        # Check if library is already processed
                        index_file = data_dir / "library_index.json"
                        video_file = data_dir / "library.mp4"
                        
                        is_processed = (
                            index_file.exists() and 
                            video_file.exists() and 
                            not self.force_reprocess
                        )
                        
                        libraries.append({
                            "id": item.name,
                            "pdf_dir": pdf_dir,
                            "data_dir": data_dir,
                            "pdf_count": len(pdf_files),
                            "is_processed": is_processed,
                            "index_file": index_file,
                            "video_file": video_file
                        })
        
        # Sort by library ID (numeric)
        libraries.sort(key=lambda x: int(x["id"]))
        return libraries
    
    def process_single_library(self, library_info: Dict[str, Any]) -> bool:
        """Process a single library instance."""
        library_id = library_info["id"]
        pdf_dir = library_info["pdf_dir"]
        data_dir = library_info["data_dir"]
        
        print(f"\n📚 Processing Library {library_id}")
        print(f"   📂 PDF Directory: {pdf_dir}")
        print(f"   📂 Data Directory: {data_dir}")
        
        # Create data directory
        data_dir.mkdir(exist_ok=True)
        
        # Initialize encoder for this library
        try:
            # Import memvid only when actually needed
            with suppress_stdout(), suppress_stderr(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from memvid import MemvidEncoder
                
            encoder = MemvidEncoder()
            
            # Apply monkey patch for parallel QR generation
            encoder = monkey_patch_parallel_qr_generation(encoder, self.encoder_workers)
        except Exception as e:
            print(f"     ❌ Failed to initialize MemvidEncoder: {e}")
            print(f"     💡 Check your memvid installation and dependencies")
            return False
        
        # Get already processed PDFs for this library
        index_path = data_dir / "library_index.json"
        processed_pdfs = get_processed_pdfs_from_index(index_path) if not self.force_reprocess else set()
        
        # Track page offsets for this library
        page_offsets = {}
        
        # Process PDFs in this library
        pdf_files = list(pdf_dir.glob("*.pdf"))
        processed_count = 0
        skipped_count = 0
        
        print(f"   📄 Found {len(pdf_files)} PDF files")
        
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"   [{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
            
            if pdf_path.name in processed_pdfs:
                print(f"     ✅ Skipped (already processed)")
                skipped_count += 1
                continue
            
            if self.process_pdf_for_library(pdf_path, encoder, page_offsets):
                processed_count += 1
        
        # Build video and index if we processed any files
        if processed_count > 0 or (skipped_count > 0 and self.force_reprocess):
            video_path = str(data_dir / "library.mp4")
            index_path = str(data_dir / "library_index.json")
            
            self.create_enhanced_index_for_library(encoder, video_path, index_path, page_offsets)
            
            print(f"   ✅ Library {library_id} completed!")
            print(f"      📊 Processed: {processed_count} new, Skipped: {skipped_count}")
            return True
        elif skipped_count > 0:
            print(f"   ✅ Library {library_id} already processed (skipped {skipped_count} files)")
            return True
        else:
            print(f"   ❌ No PDFs processed in Library {library_id}")
            return False
    
    def process_pdf_for_library(self, pdf_path: Path, encoder: 'MemvidEncoder', page_offsets: Dict) -> bool:
        """Process a single PDF for a specific library encoder."""
        try:
            # Extract text with page mapping and detect page offset
            print(f"     📄 Extracting text from {pdf_path.name}...")
            page_texts, num_pages, page_offset = self.extract_text_with_pages(pdf_path)
            
            if not page_texts:
                print(f"     ❌ Warning: No text extracted from {pdf_path.name}")
                return False
            
            # Create enhanced chunks with page offset correction
            print(f"     🔧 Creating enhanced chunks...")
            enhanced_chunks = self.create_enhanced_chunks(page_texts, page_offset)
            
            if not enhanced_chunks:
                print(f"     ❌ Warning: No chunks created from {pdf_path.name}")
                return False
            
            print(f"     📊 Pages: {num_pages}, Enhanced chunks: {len(enhanced_chunks)}")
            
            # Store page offset for this file
            page_offsets[str(pdf_path)] = page_offset
            
            # Get sample text for metadata extraction (first 10 chunks)
            sample_chunks = enhanced_chunks[:10]
            sample_text = "\n".join([chunk.text for chunk in sample_chunks])
            
            # Extract metadata using Ollama
            print(f"     🤖 Extracting metadata...")
            metadata = self.extract_metadata_with_ollama(sample_text, pdf_path.name)
            
            print(f"     📚 Title: {metadata['title'][:60]}{'...' if len(metadata['title']) > 60 else ''}")
            print(f"     👤 Authors: {metadata['authors'][:50]}{'...' if len(metadata['authors']) > 50 else ''}")
            print(f"     📅 Year: {metadata['year']}")
            
            # Add chunks to encoder with enhanced metadata
            for chunk in enhanced_chunks:
                chunk_metadata = {
                    "file_name": pdf_path.name,
                    "title": metadata["title"],
                    "authors": metadata["authors"],
                    "publishers": metadata["publishers"],
                    "year": metadata["year"],
                    "doi": metadata["doi"],
                    "num_pages": num_pages,
                    **chunk.to_dict()  # Add enhanced chunk metadata
                }
                
                # Add to encoder
                encoder.add_chunks([chunk.text])
                
                # Store metadata separately for later association
                if not hasattr(encoder, '_enhanced_metadata'):
                    encoder._enhanced_metadata = []
                encoder._enhanced_metadata.append(chunk_metadata)
            
            print(f"     ✅ Added {len(enhanced_chunks)} chunks with page references")
            return True
            
        except Exception as e:
            print(f"     ❌ Error processing {pdf_path.name}: {e}")
            return False
    
    def create_enhanced_index_for_library(self, encoder: 'MemvidEncoder', video_path: str, index_path: str, page_offsets: Dict):
        """Create enhanced index for a specific library."""
        try:
            # Build video with suppressed warnings
            print("     🎬 Building video...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = encoder.build_video(video_path, index_path)
            
            # Enhance the index with our metadata
            if hasattr(encoder, '_enhanced_metadata'):
                print("     📋 Enhancing index with detailed metadata...")
                
                # Read the basic index
                with open(index_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                
                # Enhance chunks with our metadata
                if 'metadata' in index_data and len(index_data['metadata']) == len(encoder._enhanced_metadata):
                    for i, chunk in enumerate(index_data['metadata']):
                        chunk['enhanced_metadata'] = encoder._enhanced_metadata[i]
                
                # Add summary statistics
                index_data['enhanced_stats'] = self._calculate_enhanced_stats_for_library(encoder._enhanced_metadata, page_offsets)
                
                # Write enhanced index
                with open(index_path, 'w', encoding='utf-8') as f:
                    json.dump(index_data, f, indent=2, ensure_ascii=False)
                
                print("     ✅ Enhanced index created successfully!")
            
        except Exception as e:
            print(f"     ❌ Error creating enhanced index: {e}")
    
    def _calculate_enhanced_stats_for_library(self, metadata_list: List[Dict], page_offsets: Dict) -> Dict[str, Any]:
        """Calculate enhanced statistics for a specific library."""
        if not metadata_list:
            return {}
        
        # Count by file
        files = {}
        total_pages = set()
        cross_page_chunks = 0
        
        for meta in metadata_list:
            file_name = meta.get('file_name', 'Unknown')
            start_page = meta.get('start_page', 0)
            end_page = meta.get('end_page', 0)
            
            if file_name not in files:
                files[file_name] = {
                    'chunks': 0,
                    'pages': set(),
                    'title': meta.get('title', ''),
                    'authors': meta.get('authors', ''),
                    'year': meta.get('year', '')
                }
            
            files[file_name]['chunks'] += 1
            files[file_name]['pages'].add(start_page)
            if end_page != start_page:
                files[file_name]['pages'].add(end_page)
                cross_page_chunks += 1
            
            total_pages.add(f"{file_name}:{start_page}")
        
        # Convert sets to counts
        for file_name in files:
            files[file_name]['unique_pages'] = len(files[file_name]['pages'])
            files[file_name]['pages'] = list(files[file_name]['pages'])
        
        return {
            'total_files': len(files),
            'total_chunks': len(metadata_list),
            'total_unique_pages': len(total_pages),
            'cross_page_chunks': cross_page_chunks,
            'files': files,
            'page_offsets': page_offsets
        }
        
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

    def extract_metadata_with_ollama(self, sample_text: str, filename: str = "", max_retries: int = 2) -> Dict[str, str]:
        """Enhanced metadata extraction with retries, validation and filename fallback."""
        
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
                                    print(f"     🔄 Used filename for {key}")
                                else:
                                    final_metadata[key] = ""
                            
                            print(f"     ✅ Enhanced extraction successful (attempt {attempt + 1})")
                            return final_metadata
                        else:
                            print(f"     ✅ Ollama extraction successful (attempt {attempt + 1})")
                            return ollama_metadata
                    else:
                        print(f"     ❌ Missing required keys in attempt {attempt + 1}")
                        if attempt == max_retries:
                            break
                        continue
                else:
                    print(f"     ❌ No JSON found in response (attempt {attempt + 1})")
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
    
    def _extract_year(self, year_text: str) -> str:
        """Extract first 4-digit year from text."""
        year_match = re.search(r'\b(19|20)\d{2}\b', year_text)
        return year_match.group() if year_match else ""
    
    def _empty_metadata(self) -> Dict[str, str]:
        """Return empty metadata structure."""
        return {
            "title": "",
            "authors": "",
            "publishers": "",
            "year": "",
            "doi": ""
        }
    
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
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])(\s+)([A-Z])', r'\1 \3', text)  # Fix split words like "P ackt"
        text = re.sub(r'([a-z])\s+([a-z])\s+([a-z])', lambda m: m.group(0) if len(m.group(0)) > 10 else m.group(1) + m.group(2) + m.group(3), text)
        
        # Remove artifacts and clean up
        text = text.replace('\u0000', '')  # Remove null characters
        text = text.replace('\ufffd', '')  # Remove replacement characters
        
        return text.strip()
    
    def detect_page_number_offset(self, page_texts: Dict[int, str]) -> int:
        """Always return 0 - use PDF page numbers as-is."""
        print(f"  📋 Using PDF page numbers directly (no offset)")
        return 0

    def extract_text_with_pages(self, pdf_path: Path) -> Tuple[Dict[int, str], int, int]:
        """Extract text from PDF with page-by-page mapping and detect page offset."""
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)
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
            
            doc.close()
            
            # Detect page numbering offset
            page_offset = self.detect_page_number_offset(page_texts)
            
            return page_texts, num_pages, page_offset
                
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return {}, 0, 0
    
    def create_enhanced_chunks(self, page_texts: Dict[int, str], page_offset: int = 0) -> List[EnhancedChunk]:
        """Create chunks with enhanced page metadata."""
        chunks = []
        chunk_index = 0
        
        # Sort pages by number
        sorted_pages = sorted(page_texts.keys())
        
        # Process each page
        for page_num in sorted_pages:
            page_text = page_texts[page_num]
            
            if not page_text.strip():
                continue
            
            # Split page into chunks
            page_chunks = self._split_text_into_chunks(page_text)
            
            # Create enhanced chunks for this page
            for i, chunk_text in enumerate(page_chunks):
                if chunk_text.strip():
                    # Apply page offset correction for content page numbers
                    corrected_page = max(1, page_num - page_offset)
                    enhanced_chunk = EnhancedChunk(
                        text=chunk_text,
                        start_page=corrected_page,
                        end_page=corrected_page,
                        chunk_index=chunk_index,
                        total_chunks_on_pages=len(page_chunks)
                    )
                    chunks.append(enhanced_chunk)
                    chunk_index += 1
        
        # Handle cross-page chunks (optional enhancement)
        cross_page_chunks = self._create_cross_page_chunks(page_texts, sorted_pages, page_offset)
        for cross_chunk in cross_page_chunks:
            chunks.append(cross_chunk)
            chunk_index += 1
        
        return chunks
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into token-based chunks with sliding window overlap."""
        if not text.strip():
            return []
        
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= self.chunk_size_tokens:
            # Text is smaller than chunk size, return as single chunk
            return [text]
        
        chunks = []
        start_token = 0
        
        while start_token < len(tokens):
            # Calculate end position for this chunk
            end_token = min(start_token + self.chunk_size_tokens, len(tokens))
            
            # Extract token slice for this chunk
            chunk_tokens = tokens[start_token:end_token]
            
            # Decode tokens back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Clean up the chunk text
            chunk_text = self._clean_chunk_boundaries(chunk_text)
            
            if chunk_text.strip():
                chunks.append(chunk_text)
            
            # Move start position with overlap (sliding window)
            # If this is the last chunk, break to avoid empty chunks
            if end_token >= len(tokens):
                break
                
            start_token = end_token - self.overlap_tokens
            
            # Safety check to prevent infinite loops
            if start_token < 0:
                start_token = end_token
        
        return chunks
    
    def _clean_chunk_boundaries(self, text: str) -> str:
        """Clean chunk boundaries to improve readability and semantic coherence."""
        if not text.strip():
            return ""
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # If chunk starts mid-sentence, try to find sentence beginning
        if text and not text[0].isupper() and text[0] not in '.!?':
            # Look for sentence start within first 100 characters
            sentences = re.split(r'[.!?]\s+', text)
            if len(sentences) > 1:
                # Remove the incomplete first sentence
                text = '. '.join(sentences[1:])
                if text.startswith('. '):
                    text = text[2:]
        
        # If chunk ends mid-sentence, try to complete the sentence
        if text and text[-1] not in '.!?':
            # Look for sentence end within last 100 characters
            sentence_end = re.search(r'[.!?]\s', text[-100:])
            if sentence_end:
                # Keep text up to the sentence end
                cut_point = len(text) - 100 + sentence_end.end() - 1
                text = text[:cut_point]
        
        return text.strip()
    
    def _create_cross_page_chunks(self, page_texts: Dict[int, str], 
                                sorted_pages: List[int], page_offset: int = 0) -> List[EnhancedChunk]:
        """Create chunks that span across pages for better context."""
        cross_chunks = []
        
        for i in range(len(sorted_pages) - 1):
            current_page = sorted_pages[i]
            next_page = sorted_pages[i + 1]
            
            current_text = page_texts[current_page]
            next_text = page_texts[next_page]
            
            if not current_text.strip() or not next_text.strip():
                continue
            
            # Take last 200 chars from current page and first 200 from next
            current_end = current_text[-200:].strip()
            next_start = next_text[:200].strip()
            
            if len(current_end) > 50 and len(next_start) > 50:
                cross_text = current_end + " " + next_start
                
                # Apply page offset correction for cross-page chunks
                corrected_start_page = max(1, current_page - page_offset)
                corrected_end_page = max(1, next_page - page_offset)
                
                cross_chunk = EnhancedChunk(
                    text=cross_text,
                    start_page=corrected_start_page,
                    end_page=corrected_end_page,
                    chunk_index=-1,  # Will be set later
                    total_chunks_on_pages=1
                )
                cross_chunks.append(cross_chunk)
        
        return cross_chunks
    
    
    
    def process_library(self):
        """Process all PDF libraries with enhanced metadata and multi-library support."""
        print("🔍 Discovering library instances...")
        
        libraries = self.discover_libraries()
        
        if not libraries:
            print(f"❌ No libraries found in {self.library_root}")
            print("📁 Expected structure: library/[1,2,3,...]/pdf/ with PDF files")
            print("💡 Create directories and add PDFs: mkdir -p library/1/pdf && cp *.pdf library/1/pdf/")
            return
        
        print(f"📚 Found {len(libraries)} library instances:")
        total_processed = 0
        total_skipped = 0
        
        # Display overview
        for lib in libraries:
            status = "✅ Processed" if lib["is_processed"] else "⏳ Needs processing"
            print(f"   Library {lib['id']}: {lib['pdf_count']} PDFs - {status}")
        
        print(f"\n🔧 Processing configuration:")
        print(f"   Chunk size: {self.chunk_size_tokens} tokens (~{self.chunk_size_tokens * 4} chars)")
        print(f"   Overlap: {self.overlap_tokens} tokens ({self.overlap_percentage*100:.0f}%)")
        print(f"   Chunking method: Token-based sliding window")
        print(f"   Workers: {self.encoder_workers}")
        print()
        
        # Process each library
        for library_info in libraries:
            if library_info["is_processed"]:
                print(f"\n📚 Library {library_info['id']}: Already processed (use --force-reprocess to rebuild)")
                total_skipped += library_info["pdf_count"]
                continue
            
            success = self.process_single_library(library_info)
            if success:
                total_processed += library_info["pdf_count"]
        
        # Final summary
        print(f"\n🎉 MULTI-LIBRARY PROCESSING COMPLETE!")
        print(f"📊 Summary:")
        print(f"   📚 Libraries processed: {len([lib for lib in libraries if not lib['is_processed']])}")
        print(f"   📄 Total PDFs processed: {total_processed}")
        print(f"   ⏭️  Total PDFs skipped: {total_skipped}")
        
        if total_processed > 0:
            print(f"\n💾 Generated files:")
            for lib in libraries:
                if lib["data_dir"].exists():
                    video_file = lib["data_dir"] / "library.mp4"
                    index_file = lib["data_dir"] / "library_index.json"
                    if video_file.exists() and index_file.exists():
                        print(f"   Library {lib['id']}: {lib['data_dir']}/")
                        print(f"      🎥 {video_file.name}")
                        print(f"      📋 {index_file.name}")
        
        print(f"\n💬 Next step: Run 'python3 pdf_chat.py' to chat with your libraries!")


def main():
    """Main entry point."""
    import argparse
    import sys
    import io
    
    # Aggressively suppress ALL warnings and stderr
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Additional environment variables to suppress warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['MEMVID_QUIET'] = '1'
    
    parser = argparse.ArgumentParser(description='Multi-Library PDF Processor with Skip Optimization')
    parser.add_argument('--library-root', type=str, default='./library',
                       help='Root directory containing library instances (default: ./library)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of workers for parallel processing (default: auto)')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocessing of all PDFs (ignore existing indexes)')
    
    args = parser.parse_args()
    
    processor = PDFLibraryProcessorV2(library_root=args.library_root, n_workers=args.max_workers, force_reprocess=args.force_reprocess)
    processor.process_library()


if __name__ == "__main__":
    main()