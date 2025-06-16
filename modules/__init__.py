"""
PDF Processing Modules

Modular components for PDF library processing with enhanced functionality.
Each module handles a specific aspect of the PDF-to-video pipeline.
"""

from .text_extractor import TextExtractor
from .text_chunker import TextChunker, EnhancedChunk
from .metadata_extractor import MetadataExtractor
from .embedding_service import EmbeddingService
from .qr_generator import QRGenerator
from .video_assembler import VideoAssembler

__all__ = [
    'TextExtractor',
    'TextChunker', 
    'EnhancedChunk',
    'MetadataExtractor',
    'EmbeddingService', 
    'QRGenerator',
    'VideoAssembler'
]

__version__ = "1.0.0"