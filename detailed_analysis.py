#!/usr/bin/env python3
"""
Detailed Analysis of PDF Extraction Methods
Analyzes PyPDF2, Docling, and PyMuPDF results for Issue #2 solution
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
        'issue2_specific': 0
    }
    
    # Check for specific examples from Issue #2
    issue2_patterns = [
        'Gener ative',
        'wri\^@en',
        'wri\x00en', 
        'P ackt'
    ]
    
    for pattern in issue2_patterns:
        issues['issue2_specific'] += text.count(pattern)
        
    return issues


def get_text_sample(text: str, length: int = 500) -> str:
    """Get a sample of text for comparison."""
    return text[:length].replace('\n', ' ').strip()


def analyze_three_methods():
    """Analyze all three extraction methods comprehensively."""
    
    print("🔍 DETAILED ANALYSIS: PyPDF2 vs Docling vs PyMuPDF")
    print("=" * 70)
    
    # Paths to different extraction results
    pypdf2_base = Path("extraction_test")
    docling_base = Path("extraction_test_docling") 
    multi_base = Path("extraction_test_multi")
    
    # Get available extractions
    analysis_data = []
    
    # Check what PDFs we have from all methods
    available_pdfs = set()
    
    if pypdf2_base.exists():
        for pdf_dir in pypdf2_base.iterdir():
            if pdf_dir.is_dir():
                available_pdfs.add(pdf_dir.name)
    
    if docling_base.exists():
        for pdf_dir in docling_base.iterdir():
            if pdf_dir.is_dir():
                available_pdfs.add(pdf_dir.name)
    
    if multi_base.exists():
        for method_dir in multi_base.iterdir():
            if method_dir.is_dir():
                for pdf_dir in method_dir.iterdir():
                    if pdf_dir.is_dir():
                        available_pdfs.add(pdf_dir.name)
    
    print(f"📚 Found {len(available_pdfs)} PDFs to analyze")
    
    results_summary = {
        'pypdf2': {'total_issues': 0, 'issue2_problems': 0, 'total_chars': 0, 'pdfs': 0},
        'docling': {'total_issues': 0, 'issue2_problems': 0, 'total_chars': 0, 'pdfs': 0},
        'pymupdf': {'total_issues': 0, 'issue2_problems': 0, 'total_chars': 0, 'pdfs': 0}
    }
    
    detailed_results = []
    
    for pdf_name in sorted(available_pdfs):
        print(f"\n📖 Analyzing: {pdf_name[:60]}...")
        
        pdf_analysis = {
            'pdf_name': pdf_name,
            'methods': {}
        }
        
        # Analyze PyPDF2
        pypdf2_file = pypdf2_base / pdf_name / "raw_extraction.txt"
        if pypdf2_file.exists():
            with open(pypdf2_file, 'r', encoding='utf-8') as f:
                pypdf2_text = f.read()
            
            pypdf2_issues = analyze_encoding_issues(pypdf2_text)
            total_pypdf2_issues = sum(pypdf2_issues.values())
            
            pdf_analysis['methods']['pypdf2'] = {
                'issues': pypdf2_issues,
                'total_issues': total_pypdf2_issues,
                'chars': len(pypdf2_text),
                'sample': get_text_sample(pypdf2_text)
            }
            
            results_summary['pypdf2']['total_issues'] += total_pypdf2_issues
            results_summary['pypdf2']['issue2_problems'] += pypdf2_issues['issue2_specific']
            results_summary['pypdf2']['total_chars'] += len(pypdf2_text)
            results_summary['pypdf2']['pdfs'] += 1
            
            print(f"   📊 PyPDF2: {total_pypdf2_issues} issues, {pypdf2_issues['issue2_specific']} Issue #2 problems")
        
        # Analyze Docling (try both locations)
        docling_files = [
            docling_base / pdf_name / "raw_extraction_docling.txt",
            multi_base / "docling" / pdf_name / "raw_extraction_docling.txt"
        ]
        
        docling_text = None
        for docling_file in docling_files:
            if docling_file.exists():
                with open(docling_file, 'r', encoding='utf-8') as f:
                    docling_text = f.read()
                break
        
        if docling_text:
            docling_issues = analyze_encoding_issues(docling_text)
            total_docling_issues = sum(docling_issues.values())
            
            pdf_analysis['methods']['docling'] = {
                'issues': docling_issues,
                'total_issues': total_docling_issues,
                'chars': len(docling_text),
                'sample': get_text_sample(docling_text)
            }
            
            results_summary['docling']['total_issues'] += total_docling_issues
            results_summary['docling']['issue2_problems'] += docling_issues['issue2_specific']
            results_summary['docling']['total_chars'] += len(docling_text)
            results_summary['docling']['pdfs'] += 1
            
            print(f"   📊 Docling: {total_docling_issues} issues, {docling_issues['issue2_specific']} Issue #2 problems")
        
        # Analyze PyMuPDF
        pymupdf_file = multi_base / "pymupdf" / pdf_name / "raw_extraction_pymupdf.txt"
        if pymupdf_file.exists():
            with open(pymupdf_file, 'r', encoding='utf-8') as f:
                pymupdf_text = f.read()
            
            pymupdf_issues = analyze_encoding_issues(pymupdf_text)
            total_pymupdf_issues = sum(pymupdf_issues.values())
            
            pdf_analysis['methods']['pymupdf'] = {
                'issues': pymupdf_issues,
                'total_issues': total_pymupdf_issues,
                'chars': len(pymupdf_text),
                'sample': get_text_sample(pymupdf_text)
            }
            
            results_summary['pymupdf']['total_issues'] += total_pymupdf_issues
            results_summary['pymupdf']['issue2_problems'] += pymupdf_issues['issue2_specific']
            results_summary['pymupdf']['total_chars'] += len(pymupdf_text)
            results_summary['pymupdf']['pdfs'] += 1
            
            print(f"   📊 PyMuPDF: {total_pymupdf_issues} issues, {pymupdf_issues['issue2_specific']} Issue #2 problems")
        
        detailed_results.append(pdf_analysis)
    
    # Generate comprehensive report
    print(f"\n📊 OVERALL ANALYSIS SUMMARY")
    print("=" * 50)
    
    for method, stats in results_summary.items():
        if stats['pdfs'] > 0:
            print(f"\n🔧 {method.upper()}")
            print(f"   PDFs processed: {stats['pdfs']}")
            print(f"   Total issues: {stats['total_issues']}")
            print(f"   Issue #2 problems: {stats['issue2_problems']}")
            print(f"   Total characters: {stats['total_chars']:,}")
            print(f"   Issues per PDF: {(stats['total_issues'] / stats['pdfs']):.1f}")
            if stats['total_chars'] > 0:
                print(f"   Issues per 1000 chars: {(stats['total_issues'] / stats['total_chars'] * 1000):.2f}")
    
    # Ranking
    print(f"\n🏆 METHOD RANKING")
    print("-" * 30)
    
    # Rank by total issues (lower is better)
    valid_methods = [(method, stats) for method, stats in results_summary.items() if stats['pdfs'] > 0]
    
    print(f"\n📈 By Total Issues (lower is better):")
    issue_ranking = sorted(valid_methods, key=lambda x: x[1]['total_issues'])
    for i, (method, stats) in enumerate(issue_ranking, 1):
        print(f"   {i}. {method.upper()}: {stats['total_issues']} issues")
    
    print(f"\n🎯 By Issue #2 Problems (lower is better):")
    issue2_ranking = sorted(valid_methods, key=lambda x: x[1]['issue2_problems'])
    for i, (method, stats) in enumerate(issue2_ranking, 1):
        print(f"   {i}. {method.upper()}: {stats['issue2_problems']} Issue #2 problems")
    
    print(f"\n📏 By Text Volume (higher can be better):")
    char_ranking = sorted(valid_methods, key=lambda x: x[1]['total_chars'], reverse=True)
    for i, (method, stats) in enumerate(char_ranking, 1):
        print(f"   {i}. {method.upper()}: {stats['total_chars']:,} characters")
    
    # Generate detailed report file
    report_file = Path("detailed_extraction_analysis.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE PDF EXTRACTION METHOD ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write("This report analyzes PyPDF2, Docling, and PyMuPDF for solving Issue #2\n")
        f.write("Issue #2: PDF text extraction encoding errors affecting RAG quality\n\n")
        
        f.write("📊 OVERALL STATISTICS\n")
        f.write("-" * 30 + "\n")
        for method, stats in results_summary.items():
            if stats['pdfs'] > 0:
                f.write(f"\n{method.upper()}:\n")
                f.write(f"  PDFs processed: {stats['pdfs']}\n")
                f.write(f"  Total issues: {stats['total_issues']}\n")
                f.write(f"  Issue #2 problems: {stats['issue2_problems']}\n")
                f.write(f"  Total characters: {stats['total_chars']:,}\n")
                f.write(f"  Issues per PDF: {(stats['total_issues'] / stats['pdfs']):.1f}\n")
        
        f.write(f"\n🏆 RANKINGS\n")
        f.write("-" * 20 + "\n")
        
        f.write(f"\nTotal Issues (lower=better):\n")
        for i, (method, stats) in enumerate(issue_ranking, 1):
            f.write(f"  {i}. {method.upper()}: {stats['total_issues']}\n")
        
        f.write(f"\nIssue #2 Problems (lower=better):\n")
        for i, (method, stats) in enumerate(issue2_ranking, 1):
            f.write(f"  {i}. {method.upper()}: {stats['issue2_problems']}\n")
        
        f.write(f"\n📚 PER-PDF DETAILED BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        
        for pdf_data in detailed_results:
            f.write(f"\n📖 {pdf_data['pdf_name']}\n")
            for method, data in pdf_data['methods'].items():
                f.write(f"  {method.upper()}:\n")
                f.write(f"    Total issues: {data['total_issues']}\n")
                f.write(f"    Issue #2 problems: {data['issues']['issue2_specific']}\n")
                f.write(f"    Split words: {data['issues']['split_words']}\n")
                f.write(f"    Null bytes: {data['issues']['null_bytes']}\n")
                f.write(f"    Extra spaces: {data['issues']['extra_spaces']}\n")
                f.write(f"    Characters: {data['chars']:,}\n")
                f.write(f"    Sample: {data['sample'][:100]}...\n")
    
    print(f"\n📋 Detailed report saved: {report_file}")
    
    # Final recommendations
    print(f"\n🎯 FINAL RECOMMENDATIONS FOR ISSUE #2")
    print("=" * 50)
    
    if issue2_ranking:
        best_issue2 = issue2_ranking[0][0]
        best_issue2_count = issue2_ranking[0][1]['issue2_problems']
        
        print(f"\n✅ BEST METHOD FOR ISSUE #2: {best_issue2.upper()}")
        print(f"   Issue #2 problems: {best_issue2_count}")
        
        if best_issue2_count == 0:
            print(f"   🎉 {best_issue2.upper()} COMPLETELY ELIMINATES Issue #2 problems!")
        else:
            print(f"   🔧 {best_issue2.upper()} has the fewest Issue #2 problems")
    
    if issue_ranking:
        best_overall = issue_ranking[0][0]
        best_overall_issues = issue_ranking[0][1]['total_issues']
        
        print(f"\n🏆 BEST OVERALL METHOD: {best_overall.upper()}")
        print(f"   Total issues: {best_overall_issues}")
    
    # Quality vs Quantity analysis
    print(f"\n⚖️  QUALITY vs QUANTITY ANALYSIS")
    print("-" * 40)
    
    for method, stats in results_summary.items():
        if stats['pdfs'] > 0 and stats['total_chars'] > 0:
            quality_score = 1000 - (stats['total_issues'] / stats['total_chars'] * 1000)
            print(f"   {method.upper()}: {quality_score:.1f}/1000 quality score")
            print(f"     ({stats['total_chars']:,} chars, {stats['total_issues']} issues)")


if __name__ == "__main__":
    analyze_three_methods()