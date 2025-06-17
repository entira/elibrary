import subprocess
import sys


def test_cli_help():
    result = subprocess.run([sys.executable, "pdf_processor.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
