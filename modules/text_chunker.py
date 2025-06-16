#!/usr/bin/env python3
"""
TextChunker Module

Handles intelligent text chunking with token-based sliding windows,
cross-page context preservation, and semantic boundary optimization.
"""

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import tiktoken

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


@dataclass
class EnhancedChunk:
    """Enhanced chunk with comprehensive metadata."""
    text: str
    start_page: int
    end_page: int 
    chunk_index: int
    token_count: int
    enhanced_metadata: Dict
    length: int = 0
    
    def __post_init__(self):
        """Calculate length after initialization."""
        self.length = len(self.text)


class TextChunker:
    """Advanced text chunking with token-based sliding windows."""
    
    def __init__(self, 
                 chunk_size: int = 500,
                 overlap_percentage: float = 0.15,
                 cross_page_context: int = 100):
        """Initialize TextChunker.
        
        Args:
            chunk_size: Target tokens per chunk
            overlap_percentage: Overlap between chunks (0.0-1.0)
            cross_page_context: Tokens for cross-page chunks
        """
        self.chunk_size = chunk_size
        self.overlap_size = int(chunk_size * overlap_percentage)
        self.cross_page_context = cross_page_context
        
        # Initialize tokenizer (using GPT-4 tokenizer for consistency)
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            # Fallback to cl100k_base if model-specific encoding fails
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def create_enhanced_chunks(self, page_texts: Dict[int, str], page_offset: int = 0) -> List[EnhancedChunk]:
        """Create chunks with enhanced page metadata.
        
        Args:
            page_texts: Dictionary mapping page numbers to text
            page_offset: Page numbering offset for citation accuracy
            
        Returns:
            List of EnhancedChunk objects with comprehensive metadata
        """
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
            
            for chunk_text in page_chunks:
                if not chunk_text.strip():
                    continue
                
                # Clean chunk boundaries for better readability
                cleaned_chunk = self._clean_chunk_boundaries(chunk_text)
                
                if len(cleaned_chunk.strip()) < 50:  # Skip very short chunks
                    continue
                
                # Count tokens for this chunk
                token_count = len(self.tokenizer.encode(cleaned_chunk))
                
                # Calculate display page number (with offset)
                display_page = page_num + page_offset
                
                # Create enhanced metadata
                enhanced_metadata = {
                    "file_name": "",  # Will be set by caller
                    "title": "",      # Will be set by caller  
                    "authors": "",    # Will be set by caller
                    "publishers": "", # Will be set by caller
                    "year": "",       # Will be set by caller
                    "doi": "",        # Will be set by caller
                    "chunk_index": chunk_index,
                    "page_reference": f"{display_page}",
                    "source_page": page_num,
                    "display_page": display_page,
                    "token_count": token_count,
                    "num_pages": len(sorted_pages)
                }
                
                # Create enhanced chunk
                chunk = EnhancedChunk(
                    text=cleaned_chunk,
                    start_page=page_num,
                    end_page=page_num,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    enhanced_metadata=enhanced_metadata
                )
                
                chunks.append(chunk)
                chunk_index += 1
        
        # Create cross-page chunks for better context
        cross_chunks = self._create_cross_page_chunks(page_texts, sorted_pages, page_offset)
        
        # Add cross-page chunks with updated indices
        for cross_chunk in cross_chunks:
            cross_chunk.chunk_index = chunk_index
            cross_chunk.enhanced_metadata["chunk_index"] = chunk_index
            chunks.append(cross_chunk)
            chunk_index += 1
        
        return chunks
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into token-based chunks with sliding window overlap.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Tokenize the entire text, disabling special token checks
        tokens = self.tokenizer.encode(text, disallowed_special=())
        
        if len(tokens) <= self.chunk_size:
            # Text is small enough to be a single chunk
            return [text]
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            
            # Extract tokens for this chunk
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode tokens back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append(chunk_text)
            
            # Move start index forward, considering overlap
            if end_idx >= len(tokens):
                break
                
            start_idx += self.chunk_size - self.overlap_size
        
        return chunks
    
    def _clean_chunk_boundaries(self, text: str) -> str:
        """Clean chunk boundaries to improve readability and semantic coherence.
        
        Args:
            text: Raw chunk text
            
        Returns:
            Cleaned chunk text
        """
        if not text.strip():
            return ""
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Try to end chunks at sentence boundaries when possible
        sentences = re.split(r'(?<=[.!?])\\s+', text)
        
        if len(sentences) > 1:
            # If we have multiple sentences, try to end at a complete sentence
            # But don't cut too much content
            cumulative_length = 0
            for i, sentence in enumerate(sentences):
                cumulative_length += len(sentence)
                # If we're at least 70% of original length, try to end here
                if cumulative_length >= len(text) * 0.7:
                    text = ' '.join(sentences[:i+1])
                    break
        
        # Clean up formatting issues
        text = re.sub(r'\\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\\n{3,}', '\\n\\n', text)  # Limit excessive newlines
        
        # Remove partial words at the beginning if text doesn't start with capital/number
        lines = text.split('\\n')
        if lines and lines[0] and not re.match(r'^[A-Z0-9]', lines[0].strip()):
            # Find first complete word/sentence
            words = lines[0].split()
            if len(words) > 1:
                # Find first word that starts with capital letter
                for i, word in enumerate(words):
                    if re.match(r'^[A-Z]', word):
                        lines[0] = ' '.join(words[i:])
                        break
        
        text = '\\n'.join(lines)
        
        return text.strip()
    
    def _create_cross_page_chunks(self, page_texts: Dict[int, str], 
                                sorted_pages: List[int], page_offset: int = 0) -> List[EnhancedChunk]:
        """Create chunks that span across pages for better context.
        
        Args:
            page_texts: Dictionary mapping page numbers to text
            sorted_pages: Sorted list of page numbers
            page_offset: Page numbering offset
            
        Returns:
            List of cross-page chunks
        """
        cross_chunks = []
        
        for i in range(len(sorted_pages) - 1):
            current_page = sorted_pages[i]
            next_page = sorted_pages[i + 1]
            
            current_text = page_texts[current_page].strip()
            next_text = page_texts[next_page].strip()
            
            if not current_text or not next_text:
                continue
            
            # Get ending context from current page
            current_tokens = self.tokenizer.encode(current_text)
            if len(current_tokens) >= self.cross_page_context:
                start_idx = len(current_tokens) - self.cross_page_context
                ending_tokens = current_tokens[start_idx:]
                ending_context = self.tokenizer.decode(ending_tokens)
            else:
                ending_context = current_text
            
            # Get beginning context from next page  
            next_tokens = self.tokenizer.encode(next_text)
            if len(next_tokens) >= self.cross_page_context:
                beginning_tokens = next_tokens[:self.cross_page_context]
                beginning_context = self.tokenizer.decode(beginning_tokens)
            else:
                beginning_context = next_text
            
            # Combine contexts
            cross_page_text = f"{ending_context}\\n\\n--- Page Break ---\\n\\n{beginning_context}"
            cross_page_text = self._clean_chunk_boundaries(cross_page_text)
            
            if len(cross_page_text.strip()) < 100:  # Skip very short cross-page chunks
                continue
            
            # Count tokens
            token_count = len(self.tokenizer.encode(cross_page_text))
            
            # Create display page reference
            display_current = current_page + page_offset
            display_next = next_page + page_offset
            page_reference = f"{display_current}-{display_next}"
            
            # Create enhanced metadata for cross-page chunk
            enhanced_metadata = {
                "file_name": "",  # Will be set by caller
                "title": "",      # Will be set by caller
                "authors": "",    # Will be set by caller  
                "publishers": "", # Will be set by caller
                "year": "",       # Will be set by caller
                "doi": "",        # Will be set by caller
                "chunk_index": 0, # Will be set by caller
                "page_reference": page_reference,
                "source_page": current_page,
                "display_page": display_current,
                "token_count": token_count,
                "num_pages": len(sorted_pages),
                "cross_page": True,
                "start_page": current_page,
                "end_page": next_page
            }
            
            # Create cross-page chunk
            chunk = EnhancedChunk(
                text=cross_page_text,
                start_page=current_page,
                end_page=next_page,
                chunk_index=0,  # Will be updated by caller
                token_count=token_count,
                enhanced_metadata=enhanced_metadata
            )
            
            cross_chunks.append(chunk)
        
        return cross_chunks
    
    def get_chunk_stats(self, chunks: List[EnhancedChunk]) -> Dict[str, any]:
        """Get statistics about the chunking process.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {}
        
        token_counts = [chunk.token_count for chunk in chunks]
        text_lengths = [chunk.length for chunk in chunks]
        cross_page_chunks = [chunk for chunk in chunks if chunk.enhanced_metadata.get('cross_page', False)]
        
        return {
            "total_chunks": len(chunks),
            "cross_page_chunks": len(cross_page_chunks),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "avg_chars_per_chunk": sum(text_lengths) / len(text_lengths),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "total_tokens": sum(token_counts),
            "total_characters": sum(text_lengths)
        }


# Utility functions for standalone usage
def chunk_text(page_texts: Dict[int, str], chunk_size: int = 500) -> List[EnhancedChunk]:
    """Convenience function to chunk page texts.
    
    Args:
        page_texts: Dictionary mapping page numbers to text
        chunk_size: Target tokens per chunk
        
    Returns:
        List of enhanced chunks
    """
    chunker = TextChunker(chunk_size=chunk_size)
    return chunker.create_enhanced_chunks(page_texts)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python text_chunker.py <text_file>")
        print("Example: echo 'Your text here' | python text_chunker.py -")
        sys.exit(1)
    
    # Read input
    if sys.argv[1] == "-":
        # Read from stdin
        text = sys.stdin.read()
        page_texts = {1: text}
    else:
        # Read from file
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            text = f.read()
        page_texts = {1: text}
    
    print(f"Chunking text ({len(text)} characters)...")
    
    chunker = TextChunker()
    chunks = chunker.create_enhanced_chunks(page_texts)
    stats = chunker.get_chunk_stats(chunks)
    
    print(f"\\n📊 Chunking Statistics:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Cross-page chunks: {stats['cross_page_chunks']}")
    print(f"   Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
    print(f"   Token range: {stats['min_tokens']}-{stats['max_tokens']}")
    
    # Show preview of first chunk
    if chunks:
        preview = chunks[0].text[:200]
        print(f"\\n📝 First chunk preview:\\n{preview}...")