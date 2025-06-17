#!/usr/bin/env python3
"""
VideoAssembler Module

Handles video assembly from QR frames, index creation, and metadata enhancement
with MemVid integration and comprehensive error handling.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
from io import StringIO
import contextlib
import sys

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


class VideoAssembler:
    """Video assembly and index creation with MemVid integration."""
    
    def __init__(self, 
                 fps: int = 30,
                 quality: str = "medium",
                 compression: bool = True):
        """Initialize VideoAssembler.
        
        Args:
            fps: Frames per second for video output
            quality: Video quality setting
            compression: Whether to compress video output
        """
        self.fps = fps
        self.quality = quality
        self.compression = compression
        
        # Statistics tracking
        self.stats = {
            "video_created": False,
            "index_created": False,
            "total_chunks": 0,
            "video_duration": 0.0,
            "file_size_mb": 0.0,
            "compression_ratio": 0.0
        }
    
    def assemble_video(self, 
                      frames_dir: Path, 
                      chunks: List[Dict[str, Any]], 
                      output_video: Path,
                      output_index: Path,
                      page_offsets: Optional[Dict] = None) -> Dict[str, Any]:
        """Assemble video from QR frames and create enhanced index.
        
        Args:
            frames_dir: Directory containing QR frame images
            chunks: List of chunk data with text and metadata
            output_video: Path for output video file
            output_index: Path for output index JSON file
            page_offsets: Optional page offset information
            
        Returns:
            Assembly results and statistics
        """
        try:
            # Import memvid only when needed
            with self._suppress_output():
                from memvid import MemvidEncoder
            
            print("     🎬 Assembling video from QR frames...")
            
            # Create encoder
            encoder = MemvidEncoder()
            
            # Add chunks to encoder (MemVid add_text only accepts text)
            for chunk in chunks:
                encoder.add_text(chunk.get("text", ""))
            
            # Build video with enhanced options
            with self._suppress_output():
                encoder.build_video(
                    str(output_video),
                    fps=self.fps,
                    quality=self.quality,
                    compression=self.compression
                )
            
            self.stats["video_created"] = True
            self.stats["total_chunks"] = len(chunks)
            
            # Calculate video statistics
            if output_video.exists():
                file_size = output_video.stat().st_size / (1024 * 1024)  # MB
                self.stats["file_size_mb"] = file_size
                self.stats["video_duration"] = len(chunks) / self.fps
            
            print(f"     ✅ Video created: {output_video}")
            
            # Create enhanced index
            index_result = self.create_enhanced_index(
                chunks, output_index, page_offsets
            )
            
            self.stats.update(index_result)
            
            return {
                "success": True,
                "video_path": str(output_video),
                "index_path": str(output_index),
                "stats": self.stats.copy()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stats": self.stats.copy()
            }
    
    def create_enhanced_index(self, 
                            chunks: List[Dict[str, Any]], 
                            output_path: Path,
                            page_offsets: Optional[Dict] = None) -> Dict[str, Any]:
        """Create enhanced index with comprehensive metadata.
        
        Args:
            chunks: List of chunk data
            output_path: Path for output index file
            page_offsets: Optional page offset information
            
        Returns:
            Index creation results
        """
        try:
            print("     📋 Creating enhanced index...")
            
            # Build metadata list
            metadata_list = []
            for chunk in chunks:
                chunk_metadata = {
                    "text": chunk.get("text", ""),
                    "length": len(chunk.get("text", "")),
                    "enhanced_metadata": chunk.get("metadata", {})
                }
                metadata_list.append(chunk_metadata)
            
            # Calculate enhanced statistics
            enhanced_stats = self._calculate_enhanced_stats(
                metadata_list, page_offsets or {}
            )
            
            # Build index structure
            index_data = {
                "metadata": metadata_list,
                "enhanced_stats": enhanced_stats,
                "version": "2.0",
                "created_by": "VideoAssembler",
                "assembly_config": {
                    "fps": self.fps,
                    "quality": self.quality,
                    "compression": self.compression
                }
            }
            
            # Save index
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            
            print(f"     ✅ Index created: {output_path}")
            
            return {
                "index_created": True,
                "total_entries": len(metadata_list),
                "enhanced_stats": enhanced_stats
            }
            
        except Exception as e:
            print(f"     ❌ Index creation failed: {e}")
            return {
                "index_created": False,
                "error": str(e)
            }
    
    def _calculate_enhanced_stats(self, 
                                metadata_list: List[Dict], 
                                page_offsets: Dict) -> Dict[str, Any]:
        """Calculate enhanced statistics for the index.
        
        Args:
            metadata_list: List of chunk metadata
            page_offsets: Page offset information
            
        Returns:
            Enhanced statistics dictionary
        """
        if not metadata_list:
            return {}
        
        # Count by file
        files_processed = set()
        cross_page_chunks = 0
        total_pages = 0
        
        for metadata in metadata_list:
            enhanced_meta = metadata.get("enhanced_metadata", {})
            
            # Track files
            file_name = enhanced_meta.get("file_name", "")
            if file_name:
                files_processed.add(file_name)
            
            # Count cross-page chunks
            if enhanced_meta.get("cross_page", False):
                cross_page_chunks += 1
            
            # Track pages
            num_pages = enhanced_meta.get("num_pages", 0)
            if num_pages > total_pages:
                total_pages = num_pages
        
        # Calculate text statistics
        total_text_length = sum(metadata.get("length", 0) for metadata in metadata_list)
        avg_chunk_length = total_text_length / len(metadata_list) if metadata_list else 0
        
        # Calculate token statistics
        token_counts = [
            metadata.get("enhanced_metadata", {}).get("token_count", 0) 
            for metadata in metadata_list
        ]
        total_tokens = sum(token_counts)
        avg_tokens = total_tokens / len(token_counts) if token_counts else 0
        
        return {
            "total_files": len(files_processed),
            "total_chunks": len(metadata_list),
            "cross_page_chunks": cross_page_chunks,
            "total_pages": total_pages,
            "total_text_length": total_text_length,
            "avg_chunk_length": avg_chunk_length,
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": avg_tokens,
            "files_processed": list(files_processed),
            "page_offsets": page_offsets
        }
    
    def create_library_index(self, 
                           encoder: Any, 
                           video_path: str, 
                           index_path: str, 
                           page_offsets: Dict) -> bool:
        """Create enhanced index for a specific library.
        
        Args:
            encoder: MemvidEncoder instance
            video_path: Path to video file
            index_path: Path for index file
            page_offsets: Page offset information
            
        Returns:
            True if index creation was successful
        """
        try:
            print("     🎬 Building video...")
            with self._suppress_output():
                encoder.build_video(video_path)
            
            print("     📋 Creating enhanced index...")
            
            # Get raw index data
            raw_index = encoder.get_index()
            
            # Extract chunks for enhancement
            chunks = []
            for text_data in encoder.text_data:
                chunk = {
                    "text": text_data.get("text", ""),
                    "metadata": text_data.get("metadata", {})
                }
                chunks.append(chunk)
            
            # Create enhanced index
            result = self.create_enhanced_index(
                chunks, Path(index_path), page_offsets
            )
            
            return result.get("index_created", False)
            
        except Exception as e:
            print(f"     ❌ Library index creation failed: {e}")
            return False
    
    def validate_video_output(self, video_path: Path) -> Dict[str, Any]:
        """Validate video output file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Validation results
        """
        if not video_path.exists():
            return {
                "valid": False,
                "error": "Video file does not exist"
            }
        
        file_size = video_path.stat().st_size
        if file_size == 0:
            return {
                "valid": False,
                "error": "Video file is empty"
            }
        
        # Basic format validation (check if it's actually a video)
        try:
            # Try to get basic info about the video
            # Note: This is a basic check - could be enhanced with ffprobe
            with open(video_path, 'rb') as f:
                header = f.read(8)
            
            # Check for common video file signatures
            video_signatures = [
                b'\\x00\\x00\\x00\\x18ftypmp4',  # MP4
                b'\\x00\\x00\\x00\\x20ftypmp4',  # MP4
                b'\\x1a\\x45\\xdf\\xa3',        # WebM/MKV
            ]
            
            is_video = any(header.startswith(sig[:len(header)]) for sig in video_signatures)
            
            return {
                "valid": True,
                "file_size": file_size,
                "file_size_mb": file_size / (1024 * 1024),
                "appears_to_be_video": is_video
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Could not validate video format: {e}"
            }
    
    def validate_index_output(self, index_path: Path) -> Dict[str, Any]:
        """Validate index output file.
        
        Args:
            index_path: Path to index file
            
        Returns:
            Validation results
        """
        if not index_path.exists():
            return {
                "valid": False,
                "error": "Index file does not exist"
            }
        
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # Check required fields
            required_fields = ["metadata", "enhanced_stats"]
            missing_fields = [field for field in required_fields if field not in index_data]
            
            if missing_fields:
                return {
                    "valid": False,
                    "error": f"Missing required fields: {missing_fields}"
                }
            
            metadata_count = len(index_data.get("metadata", []))
            
            return {
                "valid": True,
                "metadata_entries": metadata_count,
                "has_enhanced_stats": "enhanced_stats" in index_data,
                "version": index_data.get("version", "unknown")
            }
            
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "error": f"Invalid JSON format: {e}"
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Could not validate index: {e}"
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get assembly statistics.
        
        Returns:
            Statistics dictionary
        """
        return self.stats.copy()
    
    def _suppress_output(self):
        """Context manager to suppress stdout, stderr and warnings."""
        @contextlib.contextmanager
        def suppress_all():
            import warnings
            with warnings.catch_warnings(), \
                 contextlib.redirect_stdout(StringIO()), \
                 contextlib.redirect_stderr(StringIO()):
                warnings.simplefilter("ignore")
                yield
        return suppress_all()


# Utility functions for standalone usage
def assemble_video_from_frames(frames_dir: str, 
                              chunks_data: List[Dict], 
                              output_video: str,
                              output_index: str) -> Dict[str, Any]:
    """Convenience function to assemble video from frames.
    
    Args:
        frames_dir: Directory containing QR frames
        chunks_data: List of chunk data dictionaries
        output_video: Output video file path
        output_index: Output index file path
        
    Returns:
        Assembly results
    """
    assembler = VideoAssembler()
    return assembler.assemble_video(
        Path(frames_dir),
        chunks_data,
        Path(output_video),
        Path(output_index)
    )


if __name__ == "__main__":
    # Example usage and testing
    import sys
    import tempfile
    
    if len(sys.argv) < 2:
        print("Usage: python video_assembler.py <command> [args]")
        print("Commands:")
        print("  validate-video <video_file>     - Validate video file")
        print("  validate-index <index_file>     - Validate index file")
        print("  test-assembly                   - Test video assembly")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "validate-video" and len(sys.argv) >= 3:
        video_file = sys.argv[2]
        print(f"Validating video file: {video_file}")
        
        assembler = VideoAssembler()
        result = assembler.validate_video_output(Path(video_file))
        
        if result["valid"]:
            print("✅ Video file is valid")
            print(f"   File size: {result['file_size_mb']:.2f} MB")
            print(f"   Appears to be video: {result['appears_to_be_video']}")
        else:
            print(f"❌ Video validation failed: {result['error']}")
    
    elif command == "validate-index" and len(sys.argv) >= 3:
        index_file = sys.argv[2]
        print(f"Validating index file: {index_file}")
        
        assembler = VideoAssembler()
        result = assembler.validate_index_output(Path(index_file))
        
        if result["valid"]:
            print("✅ Index file is valid")
            print(f"   Metadata entries: {result['metadata_entries']}")
            print(f"   Has enhanced stats: {result['has_enhanced_stats']}")
            print(f"   Version: {result['version']}")
        else:
            print(f"❌ Index validation failed: {result['error']}")
    
    elif command == "test-assembly":
        print("Testing video assembly with sample data...")
        
        # Create sample chunks
        sample_chunks = [
            {
                "text": "First chunk of text for video assembly testing.",
                "metadata": {
                    "chunk_index": 0,
                    "page_reference": "1",
                    "token_count": 10
                }
            },
            {
                "text": "Second chunk with different content for testing.",
                "metadata": {
                    "chunk_index": 1,
                    "page_reference": "1",
                    "token_count": 9
                }
            }
        ]
        
        print(f"Sample chunks: {len(sample_chunks)}")
        
        # Note: This test would require actual QR frames and MemVid
        # For now, just test the index creation
        with tempfile.TemporaryDirectory() as temp_dir:
            assembler = VideoAssembler()
            index_path = Path(temp_dir) / "test_index.json"
            
            result = assembler.create_enhanced_index(sample_chunks, index_path)
            
            if result["index_created"]:
                print("✅ Index creation test passed")
                print(f"   Entries: {result['total_entries']}")
                
                # Validate the created index
                validation = assembler.validate_index_output(index_path)
                print(f"   Validation: {'PASSED' if validation['valid'] else 'FAILED'}")
            else:
                print(f"❌ Index creation test failed: {result.get('error', 'Unknown error')}")
    
    else:
        print("❌ Invalid command or missing arguments")
        sys.exit(1)
