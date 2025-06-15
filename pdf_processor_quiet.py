#!/usr/bin/env python3
"""
Quiet wrapper for PDF Library Processor - suppresses all warnings
"""

import sys
import os
import subprocess

# Set environment variables before subprocess
env = os.environ.copy()
env['PYTHONWARNINGS'] = 'ignore'
env['TF_CPP_MIN_LOG_LEVEL'] = '3'
env['TOKENIZERS_PARALLELISM'] = 'false'

# Run the original script with stderr completely redirected to null
cmd = [sys.executable, 'pdf_library_processor.py'] + sys.argv[1:]

with open(os.devnull, 'w') as devnull:
    try:
        result = subprocess.run(cmd, stderr=devnull, env=env, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n👋 Process interrupted")
        sys.exit(1)