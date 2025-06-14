#!/usr/bin/env python3
"""
PDF Extraction Comparison Tool
Compares PyPDF2 vs Docling extraction quality and analyzes Issue #2 fixes
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple


def analyze_encoding_issues(text: str) -> Dict[str, int]:
    """Analyze text for encoding issues mentioned in Issue #2."""
    issues = {
        'null_bytes': len(re.findall(r'\x00|\^@', text)),
        'split_words': len(re.findall(r'[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]', text)),
        'extra_spaces': len(re.findall(r'  +', text)),
        'unicode_errors': len(re.findall(r'\ufffd', text)),
        'broken_words': 0
    }
    
    # Check for specific examples from Issue #2
    if 'Gener ative' in text:
        issues['broken_words'] += 1
    if 'wri\^@en' in text or 'wri\x00en' in text:
        issues['broken_words'] += 1  
    if 'P ackt' in text:
        issues['broken_words'] += 1
        
    return issues


def compare_extractions():
    """Compare PyPDF2 vs Docling extraction quality."""
    extraction_test = Path("extraction_test")
    extraction_test_docling = Path("extraction_test_docling")
    
    if not extraction_test.exists() or not extraction_test_docling.exists():
        print("❌ Missing extraction directories")
        return
    
    print("🔍 COMPARING PYPDF2 vs DOCLING EXTRACTION QUALITY")
    print("=" * 60)
    
    # Find common PDFs
    pypdf2_dirs = [d for d in extraction_test.iterdir() if d.is_dir()]
    docling_dirs = [d for d in extraction_test_docling.iterdir() if d.is_dir()]
    
    common_pdfs = []
    for pypdf2_dir in pypdf2_dirs:
        pdf_name = pypdf2_dir.name.replace('_', ' ')
        for docling_dir in docling_dirs:
            if pdf_name in docling_dir.name.replace('_', ' '):
                common_pdfs.append((pypdf2_dir, docling_dir))
                break
    
    if not common_pdfs:
        print("❌ No common PDFs found for comparison")
        return
    
    print(f"📚 Found {len(common_pdfs)} PDFs to compare\n")
    
    total_pypdf2_issues = 0
    total_docling_issues = 0
    
    for pypdf2_dir, docling_dir in common_pdfs:
        print(f"📖 Analyzing: {pypdf2_dir.name[:50]}...")
        
        # Read PyPDF2 raw extraction
        pypdf2_raw_file = pypdf2_dir / "raw_extraction.txt"
        docling_raw_file = docling_dir / "raw_extraction_docling.txt"
        
        if not pypdf2_raw_file.exists() or not docling_raw_file.exists():
            print("   ❌ Missing raw extraction files")
            continue
            
        with open(pypdf2_raw_file, 'r', encoding='utf-8') as f:
            pypdf2_text = f.read()
            
        with open(docling_raw_file, 'r', encoding='utf-8') as f:
            docling_text = f.read()
        
        # Analyze issues
        pypdf2_issues = analyze_encoding_issues(pypdf2_text)
        docling_issues = analyze_encoding_issues(docling_text)
        
        pypdf2_total = sum(pypdf2_issues.values())
        docling_total = sum(docling_issues.values())
        
        total_pypdf2_issues += pypdf2_total
        total_docling_issues += docling_total
        
        print(f"   📊 PyPDF2 issues: {pypdf2_total}")
        print(f"   📊 Docling issues: {docling_total}")
        
        if pypdf2_total > docling_total:
            improvement = ((pypdf2_total - docling_total) / pypdf2_total * 100)
            print(f"   ✅ Docling improved by {improvement:.1f}%")
        elif docling_total > pypdf2_total:
            regression = ((docling_total - pypdf2_total) / pypdf2_total * 100)
            print(f"   ❌ Docling regression: {regression:.1f}%")
        else:
            print(f"   ➖ No significant difference")
        
        # Check specific Issue #2 problems
        issue2_pypdf2 = (
            pypdf2_text.count('Gener ative') + 
            pypdf2_text.count('wri\^@en') + 
            pypdf2_text.count('P ackt')
        )
        issue2_docling = (
            docling_text.count('Gener ative') +
            docling_text.count('wri\^@en') + 
            docling_text.count('P ackt')
        )
        
        if issue2_pypdf2 > 0:
            print(f"   🔍 Issue #2 problems in PyPDF2: {issue2_pypdf2}")
            print(f"   🔍 Issue #2 problems in Docling: {issue2_docling}")
            if issue2_docling < issue2_pypdf2:
                print(f"   ✅ Docling fixes Issue #2 problems!")
        
        print()
    
    print("📊 OVERALL COMPARISON SUMMARY")
    print("-" * 40)
    print(f"Total PyPDF2 issues: {total_pypdf2_issues}")
    print(f"Total Docling issues: {total_docling_issues}")
    
    if total_pypdf2_issues > total_docling_issues:
        improvement = ((total_pypdf2_issues - total_docling_issues) / total_pypdf2_issues * 100)
        print(f"✅ Overall improvement with Docling: {improvement:.1f}%")
        print(f"🎯 Docling is BETTER for solving Issue #2")
    elif total_docling_issues > total_pypdf2_issues:
        regression = ((total_docling_issues - total_pypdf2_issues) / total_pypdf2_issues * 100)
        print(f"❌ Overall regression with Docling: {regression:.1f}%")
        print(f"⚠️  PyPDF2 may be better for this use case")
    else:
        print("➖ Similar quality between both methods")
    
    # Text length comparison
    print(f"\n📏 TEXT LENGTH COMPARISON")
    print("-" * 40)
    for pypdf2_dir, docling_dir in common_pdfs[:3]:  # Show first 3
        pypdf2_raw_file = pypdf2_dir / "raw_extraction.txt"
        docling_raw_file = docling_dir / "raw_extraction_docling.txt"
        
        if pypdf2_raw_file.exists() and docling_raw_file.exists():
            pypdf2_len = len(pypdf2_raw_file.read_text(encoding='utf-8'))
            docling_len = len(docling_raw_file.read_text(encoding='utf-8'))
            
            print(f"📖 {pypdf2_dir.name[:40]}...")
            print(f"   PyPDF2: {pypdf2_len:,} chars")
            print(f"   Docling: {docling_len:,} chars")
            if docling_len > pypdf2_len:
                print(f"   ✅ Docling extracts {((docling_len - pypdf2_len) / pypdf2_len * 100):.1f}% more text")
            else:
                print(f"   ❌ PyPDF2 extracts {((pypdf2_len - docling_len) / docling_len * 100):.1f}% more text")


if __name__ == "__main__":
    compare_extractions()