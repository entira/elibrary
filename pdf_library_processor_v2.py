#!/usr/bin/env python3
"""
PDF Library Processor v2 with Enhanced Page Metadata
Processes PDF books with detailed page tracking for each chunk.
"""

import os
import json
import re
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import PyPDF2
from tqdm import tqdm
from memvid import MemvidEncoder


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
    """Enhanced PDF processor with detailed page tracking."""
    
    def __init__(self, pdf_dir: str = "./pdf_books", output_dir: str = "./memvid_out_v2"):
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize embedder and encoder
        self.embedder = OllamaEmbedder()
        self.encoder = MemvidEncoder()
        
        # Chunk configuration
        self.chunk_size = 400  # Target characters per chunk
        self.overlap = 50      # Character overlap between chunks
        
    def extract_metadata_with_ollama(self, sample_text: str) -> Dict[str, str]:
        """Extract metadata using Ollama mistral:latest model."""
        try:
            prompt = f"Extract JSON with keys: title, authors, publishers, year, doi from this text:\n\n{sample_text}\n\nReturn only valid JSON."
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral:latest",
                    "prompt": prompt,
                    "options": {"temperature": 0.1}
                },
                timeout=60
            )
            response.raise_for_status()
            
            # Parse streaming response
            response_text = ""
            for line in response.text.strip().split('\n'):
                if line:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            response_text += chunk["response"]
                    except json.JSONDecodeError:
                        continue
            
            # Try to extract JSON from response
            try:
                # Look for JSON in the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    metadata = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")
                    
                # Clean and validate metadata
                clean_metadata = {
                    "title": str(metadata.get("title", "")).strip(),
                    "authors": str(metadata.get("authors", "")).strip(),
                    "publishers": str(metadata.get("publishers", "")).strip(),
                    "year": self._extract_year(str(metadata.get("year", ""))),
                    "doi": str(metadata.get("doi", "")).strip()
                }
                
                return clean_metadata
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse JSON from Ollama response: {e}")
                return self._empty_metadata()
                
        except Exception as e:
            print(f"Error extracting metadata with Ollama: {e}")
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
    
    def extract_text_with_pages(self, pdf_path: Path) -> Tuple[Dict[int, str], int]:
        """Extract text from PDF with page-by-page mapping."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                page_texts = {}
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        page_texts[page_num] = page_text.strip()
                    except Exception as e:
                        print(f"Error extracting text from page {page_num}: {e}")
                        page_texts[page_num] = ""
                
                return page_texts, num_pages
                
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return {}, 0
    
    def create_enhanced_chunks(self, page_texts: Dict[int, str]) -> List[EnhancedChunk]:
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
                    enhanced_chunk = EnhancedChunk(
                        text=chunk_text,
                        start_page=page_num,
                        end_page=page_num,
                        chunk_index=chunk_index,
                        total_chunks_on_pages=len(page_chunks)
                    )
                    chunks.append(enhanced_chunk)
                    chunk_index += 1
        
        # Handle cross-page chunks (optional enhancement)
        cross_page_chunks = self._create_cross_page_chunks(page_texts, sorted_pages)
        for cross_chunk in cross_page_chunks:
            chunks.append(cross_chunk)
            chunk_index += 1
        
        return chunks
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find end position
            end = start + self.chunk_size
            
            # If we're not at the end, try to break at word boundary
            if end < len(text):
                # Look for word boundary within last 50 characters
                word_boundary = text.rfind(' ', max(start, end - 50), end)
                if word_boundary > start:
                    end = word_boundary
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - self.overlap)
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def _create_cross_page_chunks(self, page_texts: Dict[int, str], 
                                sorted_pages: List[int]) -> List[EnhancedChunk]:
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
                
                cross_chunk = EnhancedChunk(
                    text=cross_text,
                    start_page=current_page,
                    end_page=next_page,
                    chunk_index=-1,  # Will be set later
                    total_chunks_on_pages=1
                )
                cross_chunks.append(cross_chunk)
        
        return cross_chunks
    
    def process_pdf_enhanced(self, pdf_path: Path) -> bool:
        """Process single PDF with enhanced chunking."""
        try:
            print(f"Processing: {pdf_path.name}")
            
            # Extract text with page mapping
            page_texts, num_pages = self.extract_text_with_pages(pdf_path)
            
            if not page_texts:
                print(f"Warning: No text extracted from {pdf_path.name}")
                return False
            
            # Create enhanced chunks
            enhanced_chunks = self.create_enhanced_chunks(page_texts)
            
            if not enhanced_chunks:
                print(f"Warning: No chunks created from {pdf_path.name}")
                return False
            
            print(f"  - Pages: {num_pages}")
            print(f"  - Enhanced chunks: {len(enhanced_chunks)}")
            
            # Get sample text for metadata extraction (first 10 chunks)
            sample_chunks = enhanced_chunks[:10]
            sample_text = "\n".join([chunk.text for chunk in sample_chunks])
            
            # Extract metadata using Ollama
            print("  - Extracting metadata...")
            metadata = self.extract_metadata_with_ollama(sample_text)
            
            print(f"  - Title: {metadata['title'][:60]}{'...' if len(metadata['title']) > 60 else ''}")
            print(f"  - Authors: {metadata['authors'][:50]}{'...' if len(metadata['authors']) > 50 else ''}")
            print(f"  - Year: {metadata['year']}")
            
            # Add chunks to memvid with enhanced metadata
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
                
                # Add to encoder (we need to use add_chunks method)
                self.encoder.add_chunks([chunk.text])
                
                # Store metadata separately for later association
                if not hasattr(self.encoder, '_enhanced_metadata'):
                    self.encoder._enhanced_metadata = []
                self.encoder._enhanced_metadata.append(chunk_metadata)
            
            print(f"  - Added {len(enhanced_chunks)} chunks with page references")
            return True
            
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            return False
    
    def create_enhanced_index(self, video_path: str, index_path: str):
        """Create enhanced index with detailed metadata."""
        try:
            # Build basic video
            print("Building video...")
            self.encoder.build_video(video_path, index_path)
            
            # Enhance the index with our metadata
            if hasattr(self.encoder, '_enhanced_metadata'):
                print("Enhancing index with detailed metadata...")
                
                # Read the basic index
                with open(index_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                
                # Enhance chunks with our metadata
                if 'chunks' in index_data and len(index_data['chunks']) == len(self.encoder._enhanced_metadata):
                    for i, chunk in enumerate(index_data['chunks']):
                        chunk['metadata'] = self.encoder._enhanced_metadata[i]
                
                # Add summary statistics
                index_data['enhanced_stats'] = self._calculate_enhanced_stats()
                
                # Write enhanced index
                with open(index_path, 'w', encoding='utf-8') as f:
                    json.dump(index_data, f, indent=2, ensure_ascii=False)
                
                print("✅ Enhanced index created successfully!")
            
        except Exception as e:
            print(f"Error creating enhanced index: {e}")
    
    def _calculate_enhanced_stats(self) -> Dict[str, Any]:
        """Calculate enhanced statistics."""
        if not hasattr(self.encoder, '_enhanced_metadata'):
            return {}
        
        metadata_list = self.encoder._enhanced_metadata
        
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
            'files': files
        }
    
    def process_library(self):
        """Process entire PDF library with enhanced metadata."""
        if not self.pdf_dir.exists():
            print(f"Error: PDF directory {self.pdf_dir} does not exist!")
            return
        
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        print(f"Output directory: {self.output_dir}")
        print(f"Chunk size: {self.chunk_size} chars, Overlap: {self.overlap} chars")
        print()
        
        processed_count = 0
        
        # Process each PDF
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            if self.process_pdf_enhanced(pdf_path):
                processed_count += 1
        
        if processed_count == 0:
            print("No PDFs were successfully processed!")
            return
        
        print(f"\nBuilding enhanced video index...")
        print(f"Processed {processed_count} PDFs successfully")
        
        # Build final video and enhanced index
        try:
            video_path = str(self.output_dir / "library_v2.mp4")
            index_path = str(self.output_dir / "library_v2_index.json")
            
            self.create_enhanced_index(video_path, index_path)
            
            print(f"\n✅ SUCCESS!")
            print(f"📚 Processed {processed_count} PDF books")
            print(f"🎥 Enhanced video: {self.output_dir / 'library_v2.mp4'}")
            print(f"📋 Enhanced index: {self.output_dir / 'library_v2_index.json'}")
            print(f"📄 Each chunk includes detailed page references!")
            
        except Exception as e:
            print(f"Error building enhanced video: {e}")


def main():
    """Main entry point."""
    processor = PDFLibraryProcessorV2()
    processor.process_library()


if __name__ == "__main__":
    main()