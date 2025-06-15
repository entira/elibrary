#!/bin/bash
# Clean PDF processor without warnings

export PYTHONWARNINGS=ignore
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false

python3 pdf_library_processor.py "$@" 2>/dev/null