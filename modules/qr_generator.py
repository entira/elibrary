#!/usr/bin/env python3
"""
QRGenerator Module

Handles parallel QR code generation for text chunks with progress tracking,
temporary file management, and optimized processing for video assembly.
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import multiprocessing as mp

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def generate_single_qr_global(args):
    """Global function for QR generation to avoid pickle issues.
    
    This function runs in worker processes and must be defined at module level
    to be picklable for multiprocessing.
    
    Args:
        args: Tuple containing (chunk_data, frames_dir, chunk_index)
        
    Returns:
        Tuple of (chunk_index, success, error_message)
    """
    # Aggressively suppress ALL warnings in worker processes
    import warnings
    warnings.filterwarnings("ignore")
    import os
    import sys
    from io import StringIO
    import contextlib
    
    # Redirect stdout/stderr to suppress memvid output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    try:
        chunk_data, frames_dir, chunk_index = args
        
        # Import memvid in worker process
        from memvid import MemvidEncoder
        
        # Create temporary encoder for this chunk
        encoder = MemvidEncoder()
        
        # Add the chunk data
        encoder.add_text(
            chunk_data["text"],
            metadata=chunk_data.get("metadata", {})
        )
        
        # Generate QR frame for this chunk
        frame_path = frames_dir / f"frame_{chunk_index:06d}.png"
        
        # Build QR frame (this creates the QR code)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            encoder._build_qr_frame(str(frame_path), chunk_index)
        
        return (chunk_index, True, None)
        
    except Exception as e:
        return (chunk_index, False, str(e))
    finally:
        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class QRGenerator:
    """Parallel QR code generation for text chunks."""
    
    def __init__(self, 
                 n_workers: Optional[int] = None,
                 show_progress: bool = True):
        """Initialize QRGenerator.
        
        Args:
            n_workers: Number of worker processes (None = auto-detect)
            show_progress: Whether to show progress bars
        """
        self.n_workers = n_workers if n_workers else mp.cpu_count()
        self.show_progress = show_progress
        
        # Statistics tracking
        self.stats = {
            "total_frames": 0,
            "successful_frames": 0,
            "failed_frames": 0,
            "processing_time": 0.0,
            "frames_per_second": 0.0
        }
    
    def generate_qr_frames(self, 
                          chunks: List[Dict[str, Any]], 
                          temp_dir: Path) -> Tuple[Path, Dict[str, Any]]:
        """Generate QR frames for text chunks in parallel.
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            temp_dir: Temporary directory for frame generation
            
        Returns:
            Tuple of (frames_directory, generation_stats)
        """
        import time
        from tqdm import tqdm
        
        start_time = time.time()
        
        # Create frames directory
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        # Prepare data for parallel processing
        chunk_args = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {})
            }
            chunk_args.append((chunk_data, frames_dir, i))
        
        # Process chunks in parallel
        print(f"     ⚡ Generating {len(chunks)} QR frames using {self.n_workers} workers...")
        
        successful_frames = 0
        failed_frames = 0
        errors = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Show progress if enabled
            iterator = executor.map(generate_single_qr_global, chunk_args)
            if self.show_progress:
                iterator = tqdm(iterator, total=len(chunk_args), desc="QR Generation")
            
            for chunk_index, success, error in iterator:
                if success:
                    successful_frames += 1
                else:
                    failed_frames += 1
                    errors.append(f"Frame {chunk_index}: {error}")
        
        # Calculate statistics
        processing_time = time.time() - start_time
        frames_per_second = len(chunks) / processing_time if processing_time > 0 else 0
        
        self.stats.update({
            "total_frames": len(chunks),
            "successful_frames": successful_frames,
            "failed_frames": failed_frames,
            "processing_time": processing_time,
            "frames_per_second": frames_per_second
        })
        
        # Report results
        if failed_frames > 0:
            print(f"     ⚠️ {failed_frames} frames failed to generate")
            for error in errors[:5]:  # Show first 5 errors
                print(f"        {error}")
            if len(errors) > 5:
                print(f"        ... and {len(errors) - 5} more errors")
        
        print(f"     ✅ Generated {successful_frames}/{len(chunks)} frames in {processing_time:.1f}s")
        print(f"     📊 Rate: {frames_per_second:.1f} frames/second")
        
        return frames_dir, self.stats.copy()
    
    def generate_qr_frames_sequential(self, 
                                    chunks: List[Dict[str, Any]], 
                                    temp_dir: Path) -> Tuple[Path, Dict[str, Any]]:
        """Generate QR frames sequentially (fallback for debugging).
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            temp_dir: Temporary directory for frame generation
            
        Returns:
            Tuple of (frames_directory, generation_stats)
        """
        import time
        from tqdm import tqdm
        
        start_time = time.time()
        
        # Create frames directory
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        successful_frames = 0
        failed_frames = 0
        errors = []
        
        print(f"     🐌 Generating {len(chunks)} QR frames sequentially...")
        
        iterator = enumerate(chunks)
        if self.show_progress:
            iterator = tqdm(iterator, total=len(chunks), desc="QR Generation")
        
        for i, chunk in iterator:
            try:
                chunk_data = {
                    "text": chunk.get("text", ""),
                    "metadata": chunk.get("metadata", {})
                }
                
                # Use the global function directly
                chunk_index, success, error = generate_single_qr_global(
                    (chunk_data, frames_dir, i)
                )
                
                if success:
                    successful_frames += 1
                else:
                    failed_frames += 1
                    errors.append(f"Frame {i}: {error}")
                    
            except Exception as e:
                failed_frames += 1
                errors.append(f"Frame {i}: {e}")
        
        # Calculate statistics
        processing_time = time.time() - start_time
        frames_per_second = len(chunks) / processing_time if processing_time > 0 else 0
        
        self.stats.update({
            "total_frames": len(chunks),
            "successful_frames": successful_frames,
            "failed_frames": failed_frames,
            "processing_time": processing_time,
            "frames_per_second": frames_per_second
        })
        
        # Report results
        if failed_frames > 0:
            print(f"     ⚠️ {failed_frames} frames failed to generate")
            for error in errors[:3]:  # Show first 3 errors
                print(f"        {error}")
        
        print(f"     ✅ Generated {successful_frames}/{len(chunks)} frames in {processing_time:.1f}s")
        
        return frames_dir, self.stats.copy()
    
    def monkey_patch_memvid_encoder(self, encoder: Any) -> None:
        """Monkey patch MemvidEncoder to use parallel QR generation.
        
        Args:
            encoder: MemvidEncoder instance to patch
        """
        import types
        
        def _generate_qr_frames_parallel(self, temp_dir: Path, show_progress: bool = True) -> Path:
            """Generate QR frames in parallel using ProcessPoolExecutor."""
            
            # Extract chunks data from encoder
            chunks = []
            for i, text_data in enumerate(self.text_data):
                chunk_data = {
                    "text": text_data["text"],
                    "metadata": text_data.get("metadata", {})
                }
                chunks.append(chunk_data)
            
            # Use QRGenerator to process frames
            generator = QRGenerator(n_workers=self.n_workers, show_progress=show_progress)
            frames_dir, stats = generator.generate_qr_frames(chunks, temp_dir)
            
            return frames_dir
        
        # Patch the encoder
        encoder._generate_qr_frames_parallel = types.MethodType(
            _generate_qr_frames_parallel, encoder
        )
        
        # Set number of workers
        if not hasattr(encoder, 'n_workers'):
            encoder.n_workers = self.n_workers
    
    def validate_frames(self, frames_dir: Path) -> Dict[str, Any]:
        """Validate generated QR frames.
        
        Args:
            frames_dir: Directory containing generated frames
            
        Returns:
            Validation results dictionary
        """
        if not frames_dir.exists():
            return {
                "valid": False,
                "error": "Frames directory does not exist",
                "frame_count": 0
            }
        
        # Count frames
        frame_files = list(frames_dir.glob("frame_*.png"))
        frame_count = len(frame_files)
        
        if frame_count == 0:
            return {
                "valid": False,
                "error": "No frame files found",
                "frame_count": 0
            }
        
        # Check frame sequence
        expected_indices = set(range(frame_count))
        actual_indices = set()
        
        for frame_file in frame_files:
            try:
                # Extract index from filename: frame_000123.png
                index_str = frame_file.stem.split('_')[1]
                index = int(index_str)
                actual_indices.add(index)
            except (IndexError, ValueError):
                return {
                    "valid": False,
                    "error": f"Invalid frame filename: {frame_file.name}",
                    "frame_count": frame_count
                }
        
        missing_indices = expected_indices - actual_indices
        extra_indices = actual_indices - expected_indices
        
        if missing_indices or extra_indices:
            return {
                "valid": False,
                "error": f"Frame sequence mismatch. Missing: {missing_indices}, Extra: {extra_indices}",
                "frame_count": frame_count
            }
        
        return {
            "valid": True,
            "frame_count": frame_count,
            "frames_dir": str(frames_dir)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics.
        
        Returns:
            Statistics dictionary
        """
        return self.stats.copy()
    
    def cleanup_frames(self, frames_dir: Path) -> bool:
        """Clean up generated frames directory.
        
        Args:
            frames_dir: Directory to clean up
            
        Returns:
            True if cleanup was successful
        """
        try:
            import shutil
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            return True
        except Exception as e:
            print(f"Error cleaning up frames directory: {e}")
            return False


# Utility functions for standalone usage
def generate_qr_frames_from_texts(texts: List[str], 
                                 output_dir: str, 
                                 n_workers: Optional[int] = None) -> Dict[str, Any]:
    """Convenience function to generate QR frames from text list.
    
    Args:
        texts: List of text strings
        output_dir: Output directory for frames
        n_workers: Number of worker processes
        
    Returns:
        Generation statistics
    """
    # Convert texts to chunk format
    chunks = []
    for i, text in enumerate(texts):
        chunks.append({
            "text": text,
            "metadata": {"chunk_index": i}
        })
    
    generator = QRGenerator(n_workers=n_workers)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    frames_dir, stats = generator.generate_qr_frames(chunks, output_path)
    return stats


if __name__ == "__main__":
    # Example usage and testing
    import sys
    import tempfile
    
    if len(sys.argv) < 2:
        print("Usage: python qr_generator.py <command> [args]")
        print("Commands:")
        print("  test                      - Test QR generation with sample text")
        print("  generate <text_file>      - Generate QR frames from text file")
        print("  parallel <text_file>      - Test parallel vs sequential generation")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "test":
        print("Testing QR generation with sample text...")
        
        # Create sample chunks
        sample_texts = [
            "This is the first chunk of text for QR generation testing.",
            "Here is the second chunk with different content to encode.",
            "Finally, this is the third chunk to complete our test set."
        ]
        
        chunks = []
        for i, text in enumerate(sample_texts):
            chunks.append({
                "text": text,
                "metadata": {"chunk_index": i, "test": True}
            })
        
        # Generate QR frames
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = QRGenerator(n_workers=2)
            frames_dir, stats = generator.generate_qr_frames(chunks, Path(temp_dir))
            
            # Validate frames
            validation = generator.validate_frames(frames_dir)
            
            print(f"\\n📊 Generation Statistics:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
            
            print(f"\\n✅ Validation: {'PASSED' if validation['valid'] else 'FAILED'}")
            if not validation['valid']:
                print(f"   Error: {validation['error']}")
    
    elif command == "generate" and len(sys.argv) >= 3:
        text_file = sys.argv[2]
        print(f"Generating QR frames from: {text_file}")
        
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split content into chunks (simple line-based splitting)
            lines = content.split('\\n')
            chunks = []
            for i, line in enumerate(lines):
                if line.strip():
                    chunks.append({
                        "text": line.strip(),
                        "metadata": {"line_number": i + 1}
                    })
            
            print(f"Found {len(chunks)} text chunks")
            
            # Generate frames
            output_dir = Path("./qr_frames_output")
            generator = QRGenerator()
            frames_dir, stats = generator.generate_qr_frames(chunks, output_dir)
            
            print(f"\\n📁 Frames saved to: {frames_dir}")
            print(f"📊 Generated {stats['successful_frames']} frames")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    else:
        print("❌ Invalid command or missing arguments")
        sys.exit(1)