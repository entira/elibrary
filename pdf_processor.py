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
        EmbeddingService, QRGenerator, VideoAssembler, EnhancedChunk
    )
except ImportError as e:
    print(f"❌ Module import failed: {e}")
    print("💡 Please ensure all modules are installed and accessible")
    sys.exit(1)


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
                    base_url=self.config.ollama_base_url
                )
            else:
                self.embedding_service = None
            
            self.qr_generator = QRGenerator(
                n_workers=self.config.max_workers,
                show_progress=self.config.show_progress
            )
            
            self.video_assembler = VideoAssembler(
                fps=self.config.video_fps,
                quality=self.config.video_quality,
                compression=self.config.video_compression
            )
            
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
            
            for pdf_path in pdf_files:
                print(f"\n🔄 Processing: {pdf_path.name}")
                
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
            
            # Prepare chunk data for QR generation
            chunk_data = []
            for chunk in all_chunks:
                chunk_data.append({
                    "text": chunk.text,
                    "metadata": chunk.enhanced_metadata
                })
            
            # Generate QR frames
            print(f"\n⚡ Generating QR frames...")
            temp_dir = data_dir / "temp"
            temp_dir.mkdir(exist_ok=True)
            
            frames_dir, qr_stats = self.qr_generator.generate_qr_frames(chunk_data, temp_dir)
            
            # Assemble video
            print(f"\n🎬 Assembling video...")
            video_path = library["video_file"]
            index_path = library["index_file"]
            
            result = self.video_assembler.assemble_video(
                frames_dir, chunk_data, video_path, index_path, page_offsets
            )
            
            if result["success"]:
                print(f"     ✅ Video created: {video_path}")
                print(f"     📋 Index created: {index_path}")
                
                # Update statistics
                self.stats["libraries_processed"] += 1
                self.stats["total_files"] += len(pdf_files)
                self.stats["total_chunks"] += len(all_chunks)
                
                # Cleanup temporary files
                if temp_dir.exists():
                    import shutil
                    shutil.rmtree(temp_dir)
                
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
    
    def get_module_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics from all modules."""
        stats = {
            "processor": self.stats.copy(),
            "text_chunker": self.text_chunker.get_chunk_stats([]),
            "metadata_extractor": {},  # Would need to add stats tracking
            "qr_generator": self.qr_generator.get_statistics(),
            "video_assembler": self.video_assembler.get_statistics()
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
                base_url=config.ollama_base_url
            )
            health = embedding_service.health_check()
            if health["service_available"]:
                print("   ✅ EmbeddingService initialized and healthy")
            else:
                print(f"   ⚠️ EmbeddingService initialized but not healthy: {health['error']}")
        except Exception as e:
            print(f"   ❌ EmbeddingService failed: {e}")
            all_tests_passed = False
    
    # Test QRGenerator
    print("⚡ Testing QRGenerator...")
    try:
        qr_generator = QRGenerator(n_workers=2, show_progress=False)
        print("   ✅ QRGenerator initialized successfully")
    except Exception as e:
        print(f"   ❌ QRGenerator failed: {e}")
        all_tests_passed = False
    
    # Test VideoAssembler
    print("🎬 Testing VideoAssembler...")
    try:
        video_assembler = VideoAssembler(
            fps=config.video_fps,
            quality=config.video_quality
        )
        print("   ✅ VideoAssembler initialized successfully")
    except Exception as e:
        print(f"   ❌ VideoAssembler failed: {e}")
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