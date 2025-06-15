#!/usr/bin/env python3
"""
Quick test for progress tracking
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

# Suppress ALL output during memvid import
with suppress_stdout(), suppress_stderr(), warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from pdf_library_processor import PDFLibraryProcessorV2

def main():
    print("🧪 Testing progress tracking...")
    
    # Create processor with test directory
    processor = PDFLibraryProcessorV2(
        pdf_dir="./test_single", 
        output_dir="./test_output",
        force_reprocess=True,
        n_workers=2
    )
    
    processor.process_library()

if __name__ == "__main__":
    main()