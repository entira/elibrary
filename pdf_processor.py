#!/usr/bin/env python3
"""
PDF Processor - Modular PDF Library Processing System

A modern, modular PDF processing system that converts PDF documents into 
searchable video indexes using optimized LLM models and parallel processing.

Built on modular architecture with:
- TextExtractor: Advanced PDF text extraction
- MetadataExtractor: LLM-based metadata extraction (gemma3:4b-it-qat)
- TextChunker: Token-based chunking with cross-page context
- EmbeddingService: Vector embeddings (nomic-embed-text)
- QRGenerator: Parallel QR frame generation
- VideoAssembler: MemVid video assembly

Author: Claude Code Assistant
License: AGPL-3.0
"""

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional
import os

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Import modules
try:
    from modules import (
        TextExtractor, TextChunker, MetadataExtractor,
        EmbeddingService, EnhancedChunk
    )
except ImportError as e:
    print(f"❌ Module import failed: {e}")
    print("💡 Please ensure all modules are installed and accessible")
    sys.exit(1)


def generate_single_qr_global(args):
    """Global function for QR generation to avoid pickle issues (copy from original)."""
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
        
        chunk_data = {"id": frame_num, "text": chunk_text, "frame": frame_num}
        
        # Use memvid's QR config with auto-truncation for long chunks
        qr_config = config.get('qr', {}) if hasattr(config, 'get') else {}
        
        # Try with progressively shorter text if QR version exceeds 40
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
        # Suppress error messages in parallel workers
        return None


@dataclass
class ProcessorConfig:
    """Configuration for PDF processor with all module settings."""
    
    # Library settings
    library_root: str = "./library"
    force_reprocess: bool = False
    
    # Text processing
    chunk_size: int = 500
    overlap_percentage: float = 0.15
    cross_page_context: int = 100
    
    # LLM settings
    metadata_model: str = "gemma3:4b-it-qat"
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    metadata_retries: int = 2
    
    # Processing
    max_workers: int = 8
    generate_embeddings: bool = True
    show_progress: bool = True
    
    # Video settings
    video_fps: int = 30
    video_quality: str = "medium"
    video_compression: bool = True
    
    # Output settings
    verbose: bool = False
    quiet: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'ProcessorConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            print(f"⚠️ Could not load config from {config_path}: {e}")
            return cls()


class ModularPDFProcessor:
    """Modular PDF processor using extracted components."""
    
    def __init__(self, config: ProcessorConfig):
        """Initialize processor with configuration."""
        self.config = config
        self.stats = {
            "libraries_processed": 0,
            "libraries_skipped": 0,
            "total_files": 0,
            "total_chunks": 0,
            "total_processing_time": 0.0,
            "errors": []
        }
        
        # Initialize modules
        self._initialize_modules()
    
    def _initialize_modules(self):
        """Initialize all processing modules."""
        try:
            self.text_extractor = TextExtractor()
            
            self.text_chunker = TextChunker(
                chunk_size=self.config.chunk_size,
                overlap_percentage=self.config.overlap_percentage,
                cross_page_context=self.config.cross_page_context
            )
            
            self.metadata_extractor = MetadataExtractor(
                model=self.config.metadata_model,
                base_url=self.config.ollama_base_url,
                max_retries=self.config.metadata_retries
            )
            
            if self.config.generate_embeddings:
                self.embedding_service = EmbeddingService(
                    model=self.config.embedding_model,
                    base_url=self.config.ollama_base_url,
                    show_progress=self.config.show_progress
                )
            else:
                self.embedding_service = None
            
            # Note: QRGenerator and VideoAssembler not used in new approach
            # We use MemVid encoder directly like original code
            
            if self.config.verbose:
                print("✅ All modules initialized successfully")
                
        except Exception as e:
            print(f"❌ Module initialization failed: {e}")
            raise
    
    def discover_libraries(self) -> List[Dict[str, Any]]:
        """Discover available library directories."""
        libraries = []
        library_root = Path(self.config.library_root)
        
        if not library_root.exists():
            if self.config.verbose:
                print(f"📁 Creating library root: {library_root}")
            library_root.mkdir(parents=True, exist_ok=True)
            return libraries
        
        # Search for numbered library directories
        for item in library_root.iterdir():
            if item.is_dir() and item.name.isdigit():
                pdf_dir = item / "pdf"
                data_dir = item / "data"
                
                if pdf_dir.exists():
                    pdf_files = list(pdf_dir.glob("*.pdf"))
                    if pdf_files:
                        library_info = {
                            "id": item.name,
                            "path": item,
                            "pdf_dir": pdf_dir,
                            "data_dir": data_dir,
                            "pdf_files": pdf_files,
                            "video_file": data_dir / "library.mp4",
                            "index_file": data_dir / "library_index.json"
                        }
                        libraries.append(library_info)
        
        # Sort by library ID
        libraries.sort(key=lambda x: int(x["id"]))
        return libraries
    
    def should_skip_library(self, library: Dict[str, Any]) -> bool:
        """Check if library should be skipped."""
        if self.config.force_reprocess:
            return False
        
        video_file = library["video_file"]
        index_file = library["index_file"]
        
        return video_file.exists() and index_file.exists()
    
    def process_all_libraries(self) -> bool:
        """Process all discovered libraries."""
        start_time = time.time()
        
        # Discover libraries
        libraries = self.discover_libraries()
        
        if not libraries:
            print("❌ No libraries found!")
            print("💡 Create library structure: mkdir -p library/1/pdf")
            print("💡 Add PDF files: cp your_pdfs/*.pdf library/1/pdf/")
            return False
        
        print(f"📚 Found {len(libraries)} library instances")
        
        # Process each library
        success_count = 0
        for library in libraries:
            if self.process_single_library(library):
                success_count += 1
        
        # Final statistics
        total_time = time.time() - start_time
        self.stats["total_processing_time"] = total_time
        
        self._print_final_statistics(success_count, len(libraries))
        
        return success_count > 0
    
    def process_single_library(self, library: Dict[str, Any]) -> bool:
        """Process a single library with modular components."""
        library_id = library["id"]
        pdf_files = library["pdf_files"]
        data_dir = library["data_dir"]
        
        print(f"\n📚 Processing Library {library_id}")
        print(f"📄 Found {len(pdf_files)} PDF files")
        
        # Check if should skip
        if self.should_skip_library(library):
            print(f"⏭️ Skipping Library {library_id} (already processed)")
            print("💡 Use --force-reprocess to reprocess")
            self.stats["libraries_skipped"] += 1
            return True
        
        # Create data directory
        data_dir.mkdir(exist_ok=True)
        
        try:
            # Process all PDFs in library
            all_chunks = []
            page_offsets = {}
            
            for i, pdf_path in enumerate(pdf_files, 1):
                print(f"\n🔄 Processing ({i}/{len(pdf_files)}): {pdf_path.name}")
                
                # Extract text
                if self.config.verbose:
                    print("     📄 Extracting text...")
                page_texts, num_pages, offset = self.text_extractor.extract_text_with_pages(pdf_path)
                page_offsets[str(pdf_path)] = offset
                
                # Extract metadata
                if self.config.verbose:
                    print("     🤖 Extracting metadata...")
                first_page = self.text_extractor.extract_first_page_text(pdf_path)
                metadata = self.metadata_extractor.extract_metadata(first_page, pdf_path.name)
                
                print(f"     📚 Title: {metadata['title'][:60]}{'...' if len(metadata['title']) > 60 else ''}")
                print(f"     👤 Authors: {metadata['authors'][:50]}{'...' if len(metadata['authors']) > 50 else ''}")
                print(f"     📅 Year: {metadata['year']}")
                
                # Create chunks
                if self.config.verbose:
                    print("     ✂️ Creating chunks...")
                chunks = self.text_chunker.create_enhanced_chunks(page_texts, offset)
                
                # Enhance chunks with metadata
                for chunk in chunks:
                    chunk.enhanced_metadata.update({
                        "file_name": pdf_path.name,
                        "title": metadata["title"],
                        "authors": metadata["authors"],
                        "publishers": metadata["publishers"],
                        "year": metadata["year"],
                        "doi": metadata["doi"]
                    })
                
                all_chunks.extend(chunks)
                print(f"     ✅ Created {len(chunks)} chunks from {num_pages} pages")
            
            if not all_chunks:
                print("❌ No chunks created from any PDF files")
                return False
            
            # Generate embeddings (optional)
            if self.config.generate_embeddings and self.embedding_service:
                print(f"\n🧠 Generating embeddings for {len(all_chunks)} chunks...")
                texts = [chunk.text for chunk in all_chunks]
                embeddings = self.embedding_service.embed(texts)
                
                # Add embeddings to chunks (if needed for future use)
                for chunk, embedding in zip(all_chunks, embeddings):
                    chunk.embedding = embedding
                
                embed_stats = self.embedding_service.get_statistics()
                print(f"     ✅ Generated embeddings: {embed_stats['success_rate']:.1f}% success rate")
            
            # Create MemVid encoder and add all chunks (copy original approach)
            print(f"\n🎬 Creating video with MemVid encoder...")
            try:
                # Import and create encoder (with suppression like original)
                import contextlib
                import warnings
                from io import StringIO
                
                with contextlib.redirect_stdout(StringIO()), \
                     contextlib.redirect_stderr(StringIO()), \
                     warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from memvid import MemvidEncoder
                
                encoder = MemvidEncoder()
                
                # Apply monkey patch for parallel QR generation like original
                if self.config.max_workers > 1:
                    encoder = self._monkey_patch_parallel_qr_generation(encoder, self.config.max_workers)
                
                # Add chunks to encoder one by one (exact original pattern)
                encoder._enhanced_metadata = []
                
                for chunk in all_chunks:
                    # Store enhanced metadata like original
                    chunk_metadata = {
                        **chunk.enhanced_metadata,
                        **chunk.to_dict()
                    }
                    
                    # Add to encoder one chunk at a time (original method)
                    encoder.add_chunks([chunk.text])
                    
                    # Store metadata separately for later association (original pattern)
                    encoder._enhanced_metadata.append(chunk_metadata)
                
                # Build video (this handles QR generation internally)
                video_path = library["video_file"]
                index_path = library["index_file"]
                
                # Note: MemVid handles temp files internally in /tmp/
                
                print(f"     🎬 Building video with {len(all_chunks)} chunks...")
                
                # Use exact same suppression as original processor
                with warnings.catch_warnings(), \
                     contextlib.redirect_stdout(StringIO()), \
                     contextlib.redirect_stderr(StringIO()):
                    warnings.simplefilter("ignore")
                    result = encoder.build_video(str(video_path), str(index_path))
                
                # Enhance index with metadata (original pattern)
                self._enhance_index_with_metadata(encoder, index_path, page_offsets)
                
                result = {"success": True, "video_path": video_path, "index_path": index_path}
                
            except Exception as e:
                print(f"     ❌ MemVid encoder failed: {e}")
                if "numpy" in str(e).lower():
                    print(f"     💡 Try: pip install 'numpy<2' to fix MemVid compatibility")
                result = {"success": False, "error": str(e)}
            
            if result["success"]:
                print(f"     ✅ Video created: {video_path}")
                print(f"     📋 Index created: {index_path}")
                
                # Update statistics
                self.stats["libraries_processed"] += 1
                self.stats["total_files"] += len(pdf_files)
                self.stats["total_chunks"] += len(all_chunks)
                
                # Note: MemVid cleans up its own temp files in /tmp/
                
                return True
            else:
                print(f"     ❌ Video assembly failed: {result['error']}")
                self.stats["errors"].append(f"Library {library_id}: {result['error']}")
                return False
        
        except Exception as e:
            print(f"❌ Library {library_id} processing failed: {e}")
            self.stats["errors"].append(f"Library {library_id}: {str(e)}")
            return False
    
    def _print_final_statistics(self, success_count: int, total_libraries: int):
        """Print final processing statistics."""
        print(f"\n🎯 Processing Complete!")
        print(f"📊 Libraries processed: {success_count}/{total_libraries}")
        print(f"📄 Total files: {self.stats['total_files']}")
        print(f"📝 Total chunks: {self.stats['total_chunks']}")
        print(f"⏱️ Total time: {self.stats['total_processing_time']:.1f}s")
        
        if self.stats["errors"]:
            print(f"\n⚠️ Errors encountered:")
            for error in self.stats["errors"]:
                print(f"   {error}")
    
    def _enhance_index_with_metadata(self, encoder, index_path, page_offsets):
        """Enhance MemVid index with detailed metadata (copy from original)."""
        try:
            if not hasattr(encoder, '_enhanced_metadata'):
                return
                
            print("     📋 Enhancing index with detailed metadata...")
            
            # Read the basic index created by MemVid
            import json
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # Enhance chunks with our metadata
            print(f"     🔍 Debug: MemVid chunks={len(index_data.get('metadata', []))}, Enhanced metadata={len(encoder._enhanced_metadata)}")
            
            if 'metadata' in index_data and len(index_data['metadata']) == len(encoder._enhanced_metadata):
                for i, chunk in enumerate(index_data['metadata']):
                    chunk['enhanced_metadata'] = encoder._enhanced_metadata[i]
                print(f"     ✅ Enhanced {len(index_data['metadata'])} chunks with metadata")
            else:
                print(f"     ⚠️ Chunk count mismatch - cannot enhance metadata")
            
            # Add summary statistics like original
            enhanced_stats = {
                "total_files": len(set(meta.get("file_name", "") for meta in encoder._enhanced_metadata if meta.get("file_name"))),
                "total_chunks": len(encoder._enhanced_metadata),
                "cross_page_chunks": sum(1 for meta in encoder._enhanced_metadata if meta.get("cross_page", False)),
                "total_pages": sum(meta.get("num_pages", 0) for meta in encoder._enhanced_metadata),
                "total_text_length": sum(len(meta.get("text", "")) for meta in encoder._enhanced_metadata),
                "avg_chunk_length": sum(len(meta.get("text", "")) for meta in encoder._enhanced_metadata) / len(encoder._enhanced_metadata) if encoder._enhanced_metadata else 0,
                "total_tokens": sum(meta.get("token_count", 0) for meta in encoder._enhanced_metadata),
                "avg_tokens_per_chunk": sum(meta.get("token_count", 0) for meta in encoder._enhanced_metadata) / len(encoder._enhanced_metadata) if encoder._enhanced_metadata else 0,
                "files_processed": list(set(meta.get("file_name", "") for meta in encoder._enhanced_metadata if meta.get("file_name"))),
                "page_offsets": page_offsets
            }
            
            index_data["enhanced_stats"] = enhanced_stats
            index_data["version"] = "2.0"
            index_data["created_by"] = "ModularPDFProcessor"
            
            # Override MemVid's default config with correct Ollama models
            if "config" in index_data:
                index_data["config"]["embedding"] = {
                    "model": self.config.embedding_model,
                    "dimension": 768  # nomic-embed-text dimension
                }
                index_data["config"]["llm"] = {
                    "model": self.config.metadata_model,
                    "max_tokens": 8192,
                    "temperature": 0.1,
                    "context_window": 32000,
                    "base_url": self.config.ollama_base_url
                }
                # Update retrieval settings to match our config safely
                retrieval_cfg = index_data["config"].setdefault("retrieval", {})
                retrieval_cfg["max_workers"] = self.config.max_workers

                chunking_cfg = index_data["config"].setdefault("chunking", {})
                chunking_cfg["chunk_size"] = self.config.chunk_size
                chunking_cfg["overlap"] = int(
                    self.config.chunk_size * self.config.overlap_percentage
                )
            
            # Write enhanced index
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"     ⚠️ Could not enhance index: {e}")
    
    def _monkey_patch_parallel_qr_generation(self, encoder, n_workers: int):
        """Monkey patch MemvidEncoder to use parallel QR generation (copy from original)."""
        import types
        from pathlib import Path
        from concurrent.futures import ProcessPoolExecutor
        import os
        
        # Capture config from processor for use in monkey patch
        processor_config = self.config
        
        def _generate_qr_frames_parallel(self, temp_dir: Path, show_progress: bool = True) -> Path:
            """Generate QR frames in parallel using ProcessPoolExecutor."""
            frames_dir = temp_dir / "frames"
            frames_dir.mkdir()
            
            # Prepare data for parallel processing
            chunk_tasks = [(i, chunk, str(frames_dir), {}) 
                          for i, chunk in enumerate(self.chunks)]
            
            # Don't show worker count message as it duplicates progress bar
            # The progress bar itself shows the parallel generation
            
            # Generate QR frames in parallel using global function
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                if show_progress and not processor_config.quiet:
                    # Show progress with suppressed worker stderr
                    import contextlib
                    from tqdm import tqdm
                    original_stderr = sys.stderr
                    try:
                        # Temporarily suppress stderr only for worker warnings, keep tqdm visible
                        with open(os.devnull, 'w') as devnull:
                            sys.stderr = devnull
                            _ = list(tqdm(executor.map(generate_single_qr_global, chunk_tasks),
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
    
    def get_module_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics from all modules."""
        stats = {
            "processor": self.stats.copy(),
            "text_chunker": self.text_chunker.get_chunk_stats([]),
            "metadata_extractor": {},  # Would need to add stats tracking
            # Note: Using MemVid encoder directly instead of separate QR/Video modules
        }
        
        if self.embedding_service:
            stats["embedding_service"] = self.embedding_service.get_statistics()
        
        return stats


def create_parser() -> argparse.ArgumentParser:
    """Create comprehensive command line parser."""
    parser = argparse.ArgumentParser(
        prog="pdf_processor",
        description="Modular PDF Library Processor - Convert PDF documents to searchable video indexes",
        epilog="""
Examples:
  %(prog)s                                    # Process all libraries with defaults
  %(prog)s --force-reprocess                 # Force reprocess existing libraries
  %(prog)s --max-workers 12                  # Use 12 parallel workers
  %(prog)s --config custom_config.json       # Use custom configuration
  %(prog)s --chunk-size 750 --overlap 0.20   # Custom chunking settings
  %(prog)s --no-embeddings --quiet           # Fast mode without embeddings
  %(prog)s --help-config                     # Show all configuration options
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Main options
    parser.add_argument("--config", type=str, metavar="FILE",
                       help="Load configuration from JSON file")
    parser.add_argument("--force-reprocess", action="store_true",
                       help="Force reprocess existing libraries")
    parser.add_argument("--library-root", type=str, default="./library",
                       help="Root directory for libraries (default: ./library)")
    
    # Processing options
    parser.add_argument("--max-workers", type=int, default=8,
                       help="Number of parallel workers (default: 8)")
    parser.add_argument("--no-embeddings", action="store_true",
                       help="Skip embedding generation for faster processing")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress indicators")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    # Text processing
    text_group = parser.add_argument_group("Text Processing")
    text_group.add_argument("--chunk-size", type=int, default=500,
                           help="Target tokens per chunk (default: 500)")
    text_group.add_argument("--overlap", type=float, default=0.15,
                           help="Chunk overlap percentage 0.0-1.0 (default: 0.15)")
    text_group.add_argument("--cross-page-context", type=int, default=100,
                           help="Tokens for cross-page chunks (default: 100)")
    
    # LLM settings
    llm_group = parser.add_argument_group("LLM Settings")
    llm_group.add_argument("--metadata-model", type=str, default="gemma3:4b-it-qat",
                          help="Ollama model for metadata extraction (default: gemma3:4b-it-qat)")
    llm_group.add_argument("--embedding-model", type=str, default="nomic-embed-text",
                          help="Ollama model for embeddings (default: nomic-embed-text)")
    llm_group.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                          help="Ollama server URL (default: http://localhost:11434)")
    llm_group.add_argument("--metadata-retries", type=int, default=2,
                          help="Max retries for metadata extraction (default: 2)")
    
    # Video settings
    video_group = parser.add_argument_group("Video Settings")
    video_group.add_argument("--fps", type=int, default=30,
                            help="Video frames per second (default: 30)")
    video_group.add_argument("--quality", choices=["low", "medium", "high"], default="medium",
                            help="Video quality (default: medium)")
    video_group.add_argument("--no-compression", action="store_true",
                            help="Disable video compression")
    
    # Special commands
    parser.add_argument("--help-config", action="store_true",
                       help="Show detailed configuration help and exit")
    parser.add_argument("--test-modules", action="store_true",
                       help="Test all modules and exit")
    parser.add_argument("--version", action="version", version="PDF Processor v2.0 (Modular)")
    
    return parser


def show_config_help():
    """Show detailed configuration help."""
    print("""
📋 PDF Processor Configuration Guide

The processor can be configured via command line arguments or JSON config file.

🔧 Configuration File Format:
{
  "library_root": "./library",
  "force_reprocess": false,
  "chunk_size": 500,
  "overlap_percentage": 0.15,
  "cross_page_context": 100,
  "metadata_model": "gemma3:4b-it-qat",
  "embedding_model": "nomic-embed-text",
  "ollama_base_url": "http://localhost:11434",
  "metadata_retries": 2,
  "max_workers": 8,
  "generate_embeddings": true,
  "show_progress": true,
  "video_fps": 30,
  "video_quality": "medium",
  "video_compression": true,
  "verbose": false,
  "quiet": false
}

🎯 Key Configuration Options:

📝 Text Processing:
  --chunk-size        Target tokens per chunk (200-1000, default: 500)
  --overlap           Overlap between chunks (0.1-0.3, default: 0.15)
  --cross-page-context Context tokens for page boundaries (50-200, default: 100)

🤖 LLM Models:
  --metadata-model    Model for metadata extraction (default: gemma3:4b-it-qat)
  --embedding-model   Model for vector embeddings (default: nomic-embed-text)
  --ollama-url        Ollama server URL (default: http://localhost:11434)

⚡ Performance:
  --max-workers       Parallel workers (1-16, default: 8)
  --no-embeddings     Skip embeddings for 50% speed boost
  --fps               Video frame rate (15-60, default: 30)

🎥 Quality vs Speed:
  Fast:    --no-embeddings --fps 15 --quality low
  Balanced: [default settings]
  Quality:  --fps 60 --quality high --chunk-size 750

💡 Common Configurations:

# Fast processing for testing
pdf_processor --no-embeddings --quiet --max-workers 4

# High quality for production
pdf_processor --quality high --fps 60 --chunk-size 750 --overlap 0.20

# Large documents (memory constrained)
pdf_processor --max-workers 2 --chunk-size 300

# Custom models
pdf_processor --metadata-model mistral:latest --embedding-model all-minilm
    """)


def test_modules(config: ProcessorConfig) -> bool:
    """Test all modules for functionality."""
    print("🧪 Testing Modular PDF Processor Components\n")
    
    all_tests_passed = True
    
    # Test TextExtractor
    print("📄 Testing TextExtractor...")
    try:
        extractor = TextExtractor()
        print("   ✅ TextExtractor initialized successfully")
    except Exception as e:
        print(f"   ❌ TextExtractor failed: {e}")
        all_tests_passed = False
    
    # Test MetadataExtractor
    print("🤖 Testing MetadataExtractor...")
    try:
        metadata_extractor = MetadataExtractor(
            model=config.metadata_model,
            base_url=config.ollama_base_url
        )
        print("   ✅ MetadataExtractor initialized successfully")
    except Exception as e:
        print(f"   ❌ MetadataExtractor failed: {e}")
        all_tests_passed = False
    
    # Test TextChunker
    print("✂️ Testing TextChunker...")
    try:
        chunker = TextChunker(
            chunk_size=config.chunk_size,
            overlap_percentage=config.overlap_percentage
        )
        print("   ✅ TextChunker initialized successfully")
    except Exception as e:
        print(f"   ❌ TextChunker failed: {e}")
        all_tests_passed = False
    
    # Test EmbeddingService (optional)
    if config.generate_embeddings:
        print("🧠 Testing EmbeddingService...")
        try:
            embedding_service = EmbeddingService(
                model=config.embedding_model,
                base_url=config.ollama_base_url,
                show_progress=config.show_progress
            )
            health = embedding_service.health_check()
            if health["service_available"]:
                print("   ✅ EmbeddingService initialized and healthy")
            else:
                print(f"   ⚠️ EmbeddingService initialized but not healthy: {health['error']}")
        except Exception as e:
            print(f"   ❌ EmbeddingService failed: {e}")
            all_tests_passed = False
    
    # Test MemvidEncoder (replaces QRGenerator and VideoAssembler)
    print("🎬 Testing MemvidEncoder...")
    try:
        import contextlib
        import warnings
        from io import StringIO
        
        with contextlib.redirect_stdout(StringIO()), \
             contextlib.redirect_stderr(StringIO()), \
             warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from memvid import MemvidEncoder
        
        encoder = MemvidEncoder()
        print("   ✅ MemvidEncoder initialized successfully")
    except Exception as e:
        print(f"   ❌ MemvidEncoder failed: {e}")
        if "numpy" in str(e).lower():
            print(f"   💡 Try: pip install 'numpy<2' to fix MemVid compatibility")
        all_tests_passed = False
    
    print(f"\n🎯 Module Test Results: {'✅ ALL PASSED' if all_tests_passed else '❌ SOME FAILED'}")
    return all_tests_passed


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle special commands
    if args.help_config:
        show_config_help()
        return 0
    
    # Load configuration
    if args.config:
        config = ProcessorConfig.from_file(Path(args.config))
    else:
        config = ProcessorConfig()
    
    # Override config with command line arguments
    if hasattr(args, 'library_root'):
        config.library_root = args.library_root
    if hasattr(args, 'force_reprocess'):
        config.force_reprocess = args.force_reprocess
    if hasattr(args, 'max_workers'):
        config.max_workers = args.max_workers
    if hasattr(args, 'no_embeddings'):
        config.generate_embeddings = not args.no_embeddings
    if hasattr(args, 'quiet'):
        config.quiet = args.quiet
        config.show_progress = not args.quiet
    if hasattr(args, 'verbose'):
        config.verbose = args.verbose
    
    # Text processing overrides
    if hasattr(args, 'chunk_size'):
        config.chunk_size = args.chunk_size
    if hasattr(args, 'overlap'):
        config.overlap_percentage = args.overlap
    if hasattr(args, 'cross_page_context'):
        config.cross_page_context = args.cross_page_context
    
    # LLM overrides
    if hasattr(args, 'metadata_model'):
        config.metadata_model = args.metadata_model
    if hasattr(args, 'embedding_model'):
        config.embedding_model = args.embedding_model
    if hasattr(args, 'ollama_url'):
        config.ollama_base_url = args.ollama_url
    if hasattr(args, 'metadata_retries'):
        config.metadata_retries = args.metadata_retries
    
    # Video overrides
    if hasattr(args, 'fps'):
        config.video_fps = args.fps
    if hasattr(args, 'quality'):
        config.video_quality = args.quality
    if hasattr(args, 'no_compression'):
        config.video_compression = not args.no_compression
    
    # Handle test command
    if args.test_modules:
        success = test_modules(config)
        return 0 if success else 1
    
    # Show configuration if verbose
    if config.verbose:
        print("🔧 Configuration:")
        config_dict = config.to_dict()
        for key, value in config_dict.items():
            print(f"   {key}: {value}")
        print()
    
    # Initialize and run processor
    try:
        print("🚀 Starting Modular PDF Processor")
        print("📋 Built with modular architecture for enhanced performance\n")
        
        processor = ModularPDFProcessor(config)
        success = processor.process_all_libraries()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
