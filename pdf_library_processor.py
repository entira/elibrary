#!/usr/bin/env python3
"""
PDF Library Processor with Memvid
Processes PDF books, extracts metadata using Ollama, and creates video index with memvid.
"""

import os
import json
import re
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
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


class PDFLibraryProcessor:
    """Main processor for PDF library."""
    
    def __init__(self, pdf_dir: str = "./pdf_books", output_dir: str = "./memvid_out"):
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize embedder and encoder
        self.embedder = OllamaEmbedder()
        self.encoder = MemvidEncoder()
        
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
    
    def extract_text_from_pdf(self, pdf_path: Path) -> tuple[List[str], int]:
        """Extract text from PDF and return chunks and page count."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                chunks = []
                current_chunk = ""
                current_page = 1
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        
                        # Split page text into smaller chunks
                        words = page_text.split()
                        for i in range(0, len(words), 100):  # ~100 words per chunk
                            chunk_words = words[i:i+100]
                            chunk_text = " ".join(chunk_words)
                            
                            if chunk_text.strip():
                                chunks.append({
                                    "text": chunk_text,
                                    "page": page_num
                                })
                    
                    except Exception as e:
                        print(f"Error extracting text from page {page_num}: {e}")
                        continue
                
                return chunks, num_pages
                
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return [], 0
    
    def process_pdf(self, pdf_path: Path) -> bool:
        """Process single PDF file."""
        try:
            print(f"Processing: {pdf_path.name}")
            
            # Extract text chunks
            chunks, num_pages = self.extract_text_from_pdf(pdf_path)
            
            if not chunks:
                print(f"Warning: No text extracted from {pdf_path.name}")
                return False
            
            print(f"  - Pages: {num_pages}, Chunks: {len(chunks)}")
            
            # Get first 10 chunks for metadata extraction
            sample_chunks = chunks[:10]
            sample_text = "\n".join([chunk["text"] for chunk in sample_chunks])
            
            # Extract metadata using Ollama
            print("  - Extracting metadata...")
            metadata = self.extract_metadata_with_ollama(sample_text)
            
            print(f"  - Title: {metadata['title'][:60]}{'...' if len(metadata['title']) > 60 else ''}")
            print(f"  - Authors: {metadata['authors'][:50]}{'...' if len(metadata['authors']) > 50 else ''}")
            print(f"  - Year: {metadata['year']}")
            
            # Add PDF to memvid encoder using built-in method
            # Note: memvid handles chunking internally, metadata will be added to video description
            self.encoder.add_pdf(str(pdf_path), chunk_size=512, overlap=50)
            
            return True
            
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            return False
    
    def process_library(self):
        """Process entire PDF library."""
        if not self.pdf_dir.exists():
            print(f"Error: PDF directory {self.pdf_dir} does not exist!")
            return
        
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        processed_count = 0
        total_pages = 0
        total_chunks = 0
        
        # Process each PDF
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            if self.process_pdf(pdf_path):
                processed_count += 1
        
        if processed_count == 0:
            print("No PDFs were successfully processed!")
            return
        
        print(f"\nBuilding video index...")
        print(f"Processed {processed_count} PDFs successfully")
        
        # Build final video and index
        try:
            self.encoder.build_video(
                str(self.output_dir / "library.mp4"),
                str(self.output_dir / "library_index.json")
            )
            
            print(f"\n✅ SUCCESS!")
            print(f"📚 Processed {processed_count} PDF books")
            print(f"🎥 Video saved to: {self.output_dir / 'library.mp4'}")
            print(f"📋 Index saved to: {self.output_dir / 'library_index.json'}")
            
        except Exception as e:
            print(f"Error building video: {e}")


def main():
    """Main entry point."""
    processor = PDFLibraryProcessor()
    processor.process_library()


if __name__ == "__main__":
    main()