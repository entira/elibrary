#!/usr/bin/env python3
"""
Test script for GitHub Actions auto-fix system
Creates test issues to verify the automation works.
"""

import requests
import json
import os
import sys
from typing import Dict, Any

class GitHubAutoFixTester:
    """Test the GitHub Actions auto-fix system."""
    
    def __init__(self, repo: str, token: str):
        self.repo = repo  # format: "owner/repo"
        self.token = token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    def create_test_issue(self, issue_type: str = "bug") -> Dict[str, Any]:
        """Create a test issue for auto-fix testing."""
        
        if issue_type == "bug":
            title = "[AUTO-FIX] Test: Chunk size too small for effective RAG"
            body = """## 🐛 Bug Description
**Clear and concise description of the bug:**
The current chunk size in pdf_library_processor_v2.py is set to 400 characters, which is too small for effective RAG (Retrieval Augmented Generation). This results in insufficient context for LLM responses.

## 📍 Location
**File(s) affected:**
- [x] `pdf_library_processor_v2.py`
- [ ] `pdf_library_processor.py` 
- [ ] `pdf_chat.py`
- [ ] Other: ____

**Function/Method:**
```
PDFLibraryProcessorV2.__init__() at line 100
```

## 🔍 Current Behavior
**What is happening now:**
- Chunk size is 400 characters
- RAG responses lack sufficient context
- Search results are fragmented

## ✅ Expected Behavior  
**What should happen instead:**
- Chunk size should be 1000+ characters
- Better context preservation
- More coherent RAG responses

## 📝 Reproduction Steps
1. Run: `python3 pdf_library_processor_v2.py`
2. Observe: chunk_size = 400 in output
3. Error occurs: Poor RAG performance due to small chunks

## 🧪 Test Case
**How to verify the fix works:**
```bash
# Check that chunk_size is increased
python3 -c "import pdf_library_processor_v2; proc = pdf_library_processor_v2.PDFLibraryProcessorV2(); print(f'Chunk size: {proc.chunk_size}')"
```

## 🎯 Fix Hints
**Suggested approach (optional):**
- [x] Increase chunk_size from 400 to 1000+ characters
- [x] Adjust overlap proportionally if needed
- [ ] Fix PDF text encoding issues  
- [ ] Add enhanced metadata to index file
- [ ] Other: ____

## 📊 Priority
- [ ] 🚨 Critical (breaks core functionality)
- [x] ⚡ High (significant impact)
- [ ] 📋 Medium (nice to have)
- [ ] 🔧 Low (minor improvement)

---
**Note:** This is a test issue for GitHub Actions auto-fix system verification."""
            
            labels = ["bug", "auto-fix", "test"]
            
        elif issue_type == "enhancement":
            title = "[AUTO-FIX] Enhancement: Add better error handling to PDF processor"
            body = """## 💡 Enhancement Description
**Clear description of the proposed improvement:**
Add comprehensive error handling and logging to the PDF processing pipeline to make debugging easier and improve reliability.

## 📍 Target Location
**File(s) to modify:**
- [x] `pdf_library_processor_v2.py`
- [ ] `pdf_library_processor.py`
- [ ] `pdf_chat.py`
- [ ] New file: ____

## 🎯 Goal
**What this enhancement will achieve:**
- Better error messages for failed PDF processing
- Graceful handling of corrupted PDF files
- Detailed logging for debugging purposes

## 📋 Requirements
**Specific requirements for the implementation:**
- [x] Maintain backward compatibility
- [x] Add error handling
- [x] Include logging/debugging
- [ ] Update documentation
- [ ] Add tests

## 🧪 Acceptance Criteria
**How to verify the enhancement works:**
```bash
# Test with a corrupted PDF
python3 pdf_library_processor_v2.py
# Should show clear error messages instead of crashing
```

## 📖 Implementation Hints
**Suggested approach (optional):**
```python
# Example code structure
try:
    # PDF processing code
    pass
except PyPDF2.errors.PdfReadError as e:
    logger.error(f"Failed to read PDF {pdf_path}: {e}")
    return False
except Exception as e:
    logger.error(f"Unexpected error processing {pdf_path}: {e}")
    return False
```

## 📊 Priority  
- [ ] 🚨 Critical (urgent need)
- [x] ⚡ High (important improvement)
- [ ] 📋 Medium (nice to have)
- [ ] 🔧 Low (minor enhancement)

## 🔗 Related Issues
**Link to related issues/PRs:**
- Related to PDF processing reliability

---
**Note:** This is a test enhancement for GitHub Actions auto-fix system verification."""
            
            labels = ["enhancement", "auto-fix", "test"]
        
        else:
            raise ValueError(f"Unknown issue type: {issue_type}")
        
        # Create the issue
        url = f"{self.base_url}/repos/{self.repo}/issues"
        data = {
            "title": title,
            "body": body,
            "labels": labels
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        
        return response.json()
    
    def get_issue_status(self, issue_number: int) -> Dict[str, Any]:
        """Get the status of an issue."""
        url = f"{self.base_url}/repos/{self.repo}/issues/{issue_number}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def list_pulls_for_issue(self, issue_number: int) -> list:
        """List pull requests that reference an issue."""
        url = f"{self.base_url}/repos/{self.repo}/pulls"
        params = {"state": "all", "sort": "created", "direction": "desc"}
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        
        pulls = response.json()
        related_pulls = []
        
        for pull in pulls:
            if f"#{issue_number}" in pull.get("body", "") or f"#{issue_number}" in pull.get("title", ""):
                related_pulls.append(pull)
        
        return related_pulls
    
    def run_test_suite(self):
        """Run the complete test suite."""
        print("🧪 Starting GitHub Actions Auto-Fix Test Suite")
        print(f"📍 Repository: {self.repo}")
        print()
        
        # Test 1: Bug fix
        print("🐛 Test 1: Creating bug fix issue...")
        try:
            bug_issue = self.create_test_issue("bug")
            print(f"✅ Bug issue created: #{bug_issue['number']}")
            print(f"🔗 URL: {bug_issue['html_url']}")
        except Exception as e:
            print(f"❌ Failed to create bug issue: {e}")
            return False
        
        print()
        
        # Test 2: Enhancement
        print("🚀 Test 2: Creating enhancement issue...")
        try:
            enhancement_issue = self.create_test_issue("enhancement")
            print(f"✅ Enhancement issue created: #{enhancement_issue['number']}")
            print(f"🔗 URL: {enhancement_issue['html_url']}")
        except Exception as e:
            print(f"❌ Failed to create enhancement issue: {e}")
            return False
        
        print()
        print("🎯 Test Issues Created Successfully!")
        print()
        print("📋 Next Steps:")
        print("1. Check GitHub Actions tab for workflow runs")
        print("2. Monitor issues for AI comments")
        print("3. Review generated pull requests")
        print("4. Verify that fixes work as expected")
        print()
        print("⏱️  Expected timeline:")
        print("   - Workflow start: ~30 seconds")
        print("   - AI analysis: ~1-2 minutes")
        print("   - PR creation: ~30 seconds")
        print("   - Total: ~3-5 minutes")
        
        return True


def main():
    """Main function."""
    # Get configuration from environment
    repo = os.getenv("GITHUB_REPOSITORY")
    token = os.getenv("GITHUB_TOKEN")
    
    if not repo:
        print("❌ GITHUB_REPOSITORY environment variable not set")
        print("💡 Set it like: export GITHUB_REPOSITORY='username/repo-name'")
        sys.exit(1)
    
    if not token:
        print("❌ GITHUB_TOKEN environment variable not set")
        print("💡 Create a personal access token at: https://github.com/settings/tokens")
        print("💡 Set it like: export GITHUB_TOKEN='ghp_your_token_here'")
        sys.exit(1)
    
    # Run tests
    tester = GitHubAutoFixTester(repo, token)
    success = tester.run_test_suite()
    
    if success:
        print("🎉 Test suite completed successfully!")
        sys.exit(0)
    else:
        print("💥 Test suite failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()