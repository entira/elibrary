#!/usr/bin/env python3
"""
Multi-Library PDF Text Extraction Quality Tester
Tests PyPDF2, Docling, and PyMuPDF extraction quality with configurable methods
"""

import re
import sys
from pathlib import Path
from typing import Dict, List
import PyPDF2

# Import libraries with fallbacks
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("⚠️  Docling not available")

try:
    import pymupdf as fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("⚠️  PyMuPDF not available")


class MultiPDFTester:
    """Test PDF text extraction quality using multiple libraries."""
    
    def __init__(self, pdf_dir: str = "./pdf_books", base_output_dir: str = "./extraction_test_multi"):
        self.pdf_dir = Path(pdf_dir)
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Initialize converters
        self.methods = {}
        if DOCLING_AVAILABLE:
            self.methods["docling"] = DocumentConverter()
            print("✅ Docling available")
        if PYMUPDF_AVAILABLE:
            self.methods["pymupdf"] = None  # PyMuPDF doesn't need persistent object
            print("✅ PyMuPDF available")
        
        # PyPDF2 is always available (imported above)
        self.methods["pypdf2"] = None
        print("✅ PyPDF2 available")
        
        print(f"🔧 Available extraction methods: {list(self.methods.keys())}")
    
    def clean_extracted_text(self, text: str) -> str:
        """Clean extracted PDF text from encoding issues (same as processor)."""
        if not text:
            return ""
        
        # Remove null bytes and other control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        # Fix multiple spaces while preserving intentional formatting
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'[ \t]*\n[ \t]*', '\n', text)  # Clean line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit multiple newlines
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])(\s+)([A-Z])', r'\1 \3', text)  # Fix split words like "P ackt"
        text = re.sub(r'([a-z])\s+([a-z])\s+([a-z])', lambda m: m.group(0) if len(m.group(0)) > 10 else m.group(1) + m.group(2) + m.group(3), text)
        
        # Remove artifacts and clean up
        text = text.replace('\u0000', '')  # Remove null characters
        text = text.replace('\ufffd', '')  # Remove replacement characters
        
        return text.strip()
    
    def extract_with_pypdf2(self, pdf_path: Path) -> Dict[str, str]:
        """Extract text using PyPDF2."""
        try:
            print(f"   📄 Processing with PyPDF2...")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                raw_text = ""
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        raw_text += page_text + "\n"
                        
                        if page_num % 50 == 0:
                            print(f"      📖 Processed {page_num}/{num_pages} pages")
                    except Exception as e:
                        print(f"      ❌ Error on page {page_num}: {e}")
                
                cleaned_text = self.clean_extracted_text(raw_text)
                
                return {
                    'raw': raw_text,
                    'cleaned': cleaned_text,
                    'pages': num_pages,
                    'metadata': {'title': '', 'author': '', 'subject': ''}
                }
                
        except Exception as e:
            print(f"      ❌ Error with PyPDF2: {e}")
            return {'raw': '', 'cleaned': '', 'pages': 0, 'metadata': {}}
    
    def extract_with_docling(self, pdf_path: Path) -> Dict[str, str]:
        """Extract text using Docling."""
        if not DOCLING_AVAILABLE:
            return {'raw': '', 'cleaned': '', 'pages': 0, 'metadata': {}}
            
        try:
            print(f"   📄 Processing with Docling...")
            
            result = self.methods["docling"].convert(str(pdf_path))
            raw_text = result.document.export_to_text()
            pages = getattr(result.document, 'page_count', 'unknown')
            metadata = {
                'title': getattr(result.document, 'title', ''),  
                'author': getattr(result.document, 'author', ''),
                'subject': getattr(result.document, 'subject', '')
            }
            
            cleaned_text = self.clean_extracted_text(raw_text)
            
            return {
                'raw': raw_text,
                'cleaned': cleaned_text,
                'pages': pages,
                'metadata': metadata
            }
                
        except Exception as e:
            print(f"      ❌ Error with Docling: {e}")
            return {'raw': '', 'cleaned': '', 'pages': 0, 'metadata': {}}
    
    def extract_with_pymupdf(self, pdf_path: Path) -> Dict[str, str]:
        """Extract text using PyMuPDF."""
        if not PYMUPDF_AVAILABLE:
            return {'raw': '', 'cleaned': '', 'pages': 0, 'metadata': {}}
            
        try:
            print(f"   📄 Processing with PyMuPDF...")
            
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            raw_text = ""
            
            for page_num in range(num_pages):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    raw_text += page_text + "\n"
                    
                    if (page_num + 1) % 50 == 0:
                        print(f"      📖 Processed {page_num + 1}/{num_pages} pages")
                        
                except Exception as e:
                    print(f"      ❌ Error on page {page_num + 1}: {e}")
            
            # Get metadata
            metadata = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', '')
            }
            
            doc.close()
            cleaned_text = self.clean_extracted_text(raw_text)
            
            return {
                'raw': raw_text,
                'cleaned': cleaned_text,
                'pages': num_pages,
                'metadata': metadata
            }
                
        except Exception as e:
            print(f"      ❌ Error with PyMuPDF: {e}")
            return {'raw': '', 'cleaned': '', 'pages': 0, 'metadata': {}}
    
    def analyze_text_quality(self, text_data: Dict[str, str], pdf_name: str, method: str) -> Dict[str, any]:
        """Analyze extraction quality and identify issues."""
        raw_text = text_data['raw']
        cleaned_text = text_data['cleaned']
        
        analysis = {
            'pdf_name': pdf_name,
            'method': method,
            'pages': text_data['pages'],
            'issues': {
                'null_bytes': 0,
                'split_words': 0,
                'extra_spaces': 0,
                'unicode_errors': 0,
                'empty_content': 0,
                'issue2_specific': 0  # Track specific Issue #2 problems
            },
            'examples': [],
            'stats': {
                'total_chars_raw': len(raw_text),
                'total_chars_cleaned': len(cleaned_text),
                'chars_removed': len(raw_text) - len(cleaned_text),
                'cleaning_impact': f"{((len(raw_text) - len(cleaned_text)) / len(raw_text) * 100):.1f}%" if raw_text else "0.0%"
            },
            'metadata': text_data['metadata']
        }
        
        # Check for various issues
        if not raw_text.strip():
            analysis['issues']['empty_content'] = 1
        
        if '\u0000' in raw_text or '\^@' in raw_text:
            analysis['issues']['null_bytes'] = len(re.findall(r'\u0000|\^@', raw_text))
            analysis['examples'].append(f"Null bytes found: {analysis['issues']['null_bytes']} occurrences")
        
        # Look for split words patterns
        split_word_matches = re.findall(r'[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]', raw_text)
        if split_word_matches:
            analysis['issues']['split_words'] = len(split_word_matches)
            analysis['examples'].append(f"Split words found: {split_word_matches[:5]}")
        
        extra_space_matches = re.findall(r'  +', raw_text)
        if extra_space_matches:
            analysis['issues']['extra_spaces'] = len(extra_space_matches)
        
        if '\ufffd' in raw_text:
            analysis['issues']['unicode_errors'] = len(re.findall(r'\ufffd', raw_text))
            analysis['examples'].append("Unicode replacement chars found")
        
        # Check for specific Issue #2 problems
        issue2_count = (
            raw_text.count('Gener ative') + 
            raw_text.count('wri\^@en') + 
            raw_text.count('wri\u0000en') +
            raw_text.count('P ackt')
        )
        analysis['issues']['issue2_specific'] = issue2_count
        if issue2_count > 0:
            analysis['examples'].append(f"Issue #2 specific problems: {issue2_count}")
        
        return analysis
    
    def save_extracted_text(self, text_data: Dict[str, str], pdf_name: str, method: str):
        """Save extracted text to files for manual inspection."""
        # Create method-specific subdirectory
        method_dir = self.base_output_dir / method
        method_dir.mkdir(exist_ok=True)
        
        # Create subdirectory for this PDF
        pdf_dir = method_dir / pdf_name.replace('.pdf', '')
        pdf_dir.mkdir(exist_ok=True)
        
        # Save files
        raw_file = pdf_dir / f"raw_extraction_{method}.txt"
        cleaned_file = pdf_dir / f"cleaned_extraction_{method}.txt"
        metadata_file = pdf_dir / f"metadata_{method}.txt"
        
        with open(raw_file, 'w', encoding='utf-8') as f:
            f.write(f"Raw {method.upper()} extraction from: {pdf_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(text_data['raw'])
        
        with open(cleaned_file, 'w', encoding='utf-8') as f:
            f.write(f"Cleaned {method.upper()} extraction from: {pdf_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(text_data['cleaned'])
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(f"{method.upper()} Metadata for: {pdf_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Pages: {text_data['pages']}\n")
            for key, value in text_data['metadata'].items():
                f.write(f"{key.title()}: {value}\n")
    
    def test_single_pdf_all_methods(self, pdf_path: Path) -> Dict[str, Dict]:
        """Test single PDF with all available methods."""
        print(f"📖 Processing: {pdf_path.name}")
        
        results = {}
        
        # Test PyPDF2
        if "pypdf2" in self.methods:
            text_data = self.extract_with_pypdf2(pdf_path)
            if text_data['raw']:
                analysis = self.analyze_text_quality(text_data, pdf_path.name, "pypdf2")
                self.save_extracted_text(text_data, pdf_path.name, "pypdf2")
                results["pypdf2"] = analysis
                print(f"   ✅ PyPDF2: {analysis['pages']} pages, {sum(analysis['issues'].values())} issues")
        
        # Test Docling
        if "docling" in self.methods:
            text_data = self.extract_with_docling(pdf_path)
            if text_data['raw']:
                analysis = self.analyze_text_quality(text_data, pdf_path.name, "docling")
                self.save_extracted_text(text_data, pdf_path.name, "docling")
                results["docling"] = analysis
                print(f"   ✅ Docling: {analysis['pages']} pages, {sum(analysis['issues'].values())} issues")
        
        # Test PyMuPDF
        if "pymupdf" in self.methods:
            text_data = self.extract_with_pymupdf(pdf_path)
            if text_data['raw']:
                analysis = self.analyze_text_quality(text_data, pdf_path.name, "pymupdf")
                self.save_extracted_text(text_data, pdf_path.name, "pymupdf")
                results["pymupdf"] = analysis
                print(f"   ✅ PyMuPDF: {analysis['pages']} pages, {sum(analysis['issues'].values())} issues")
        
        return results
    
    def test_all_pdfs(self):
        """Test extraction quality for all PDFs using all available methods."""
        if not self.pdf_dir.exists():
            print(f"❌ PDF directory {self.pdf_dir} does not exist!")
            return
        
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {self.pdf_dir}")
            return
        
        print(f"🔍 Testing PDF extraction quality with multiple methods")
        print(f"📁 PDF directory: {self.pdf_dir}")
        print(f"💾 Output directory: {self.base_output_dir}")
        print(f"📚 Found {len(pdf_files)} PDF files")
        print(f"🔧 Methods: {list(self.methods.keys())}")
        print()
        
        all_results = []
        
        for pdf_path in pdf_files:
            pdf_results = self.test_single_pdf_all_methods(pdf_path)
            all_results.append(pdf_results)
            print()
        
        # Generate comprehensive comparison report
        self.generate_comparison_report(all_results)
    
    def generate_comparison_report(self, all_results: List[Dict]):
        """Generate comprehensive comparison report across all methods."""
        report_file = self.base_output_dir / "multi_method_comparison_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("MULTI-METHOD PDF TEXT EXTRACTION COMPARISON REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated for {len(all_results)} PDF files\n")
            f.write(f"Methods tested: {list(self.methods.keys())}\n\n")
            
            # Overall statistics by method
            method_stats = {}
            for method in self.methods.keys():
                method_stats[method] = {
                    'total_issues': 0,
                    'total_pages': 0,
                    'total_chars': 0,
                    'issue2_problems': 0,
                    'pdfs_processed': 0
                }
            
            # Collect stats
            for pdf_results in all_results:
                for method, analysis in pdf_results.items():
                    if method in method_stats:
                        method_stats[method]['total_issues'] += sum(analysis['issues'].values())
                        method_stats[method]['total_pages'] += int(analysis['pages']) if str(analysis['pages']).isdigit() else 0
                        method_stats[method]['total_chars'] += analysis['stats']['total_chars_raw']
                        method_stats[method]['issue2_problems'] += analysis['issues']['issue2_specific']
                        method_stats[method]['pdfs_processed'] += 1
            
            # Write overall comparison
            f.write("📊 OVERALL STATISTICS BY METHOD\n")
            f.write("-" * 40 + "\n")
            
            for method, stats in method_stats.items():
                if stats['pdfs_processed'] > 0:
                    f.write(f"\n🔧 {method.upper()}\n")
                    f.write(f"   PDFs processed: {stats['pdfs_processed']}\n")
                    f.write(f"   Total pages: {stats['total_pages']}\n")
                    f.write(f"   Total issues: {stats['total_issues']}\n")
                    f.write(f"   Issue #2 problems: {stats['issue2_problems']}\n")
                    f.write(f"   Total characters: {stats['total_chars']:,}\n")
                    if stats['total_pages'] > 0:
                        f.write(f"   Issues per page: {(stats['total_issues'] / stats['total_pages']):.2f}\n")
            
            # Ranking
            f.write(f"\n🏆 METHOD RANKING (by total issues - lower is better)\n")
            f.write("-" * 50 + "\n")
            
            ranking = sorted([(method, stats['total_issues']) for method, stats in method_stats.items() if stats['pdfs_processed'] > 0], 
                           key=lambda x: x[1])
            
            for i, (method, total_issues) in enumerate(ranking, 1):
                f.write(f"{i}. {method.upper()}: {total_issues} total issues\n")
            
            # Issue #2 specific analysis
            f.write(f"\n🎯 ISSUE #2 ANALYSIS\n")
            f.write("-" * 30 + "\n")
            
            issue2_ranking = sorted([(method, stats['issue2_problems']) for method, stats in method_stats.items() if stats['pdfs_processed'] > 0], 
                                  key=lambda x: x[1])
            
            for i, (method, issue2_count) in enumerate(issue2_ranking, 1):
                f.write(f"{i}. {method.upper()}: {issue2_count} Issue #2 problems\n")
            
            if issue2_ranking:
                best_method = issue2_ranking[0][0]
                f.write(f"\n✅ BEST METHOD FOR ISSUE #2: {best_method.upper()}\n")
            
            # Per-PDF detailed breakdown
            f.write(f"\n📚 PER-PDF DETAILED ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            for i, pdf_results in enumerate(all_results):
                if pdf_results:
                    first_result = next(iter(pdf_results.values()))
                    pdf_name = first_result['pdf_name']
                    f.write(f"\n📖 {pdf_name}\n")
                    
                    for method, analysis in pdf_results.items():
                        f.write(f"   {method.upper()}:\n")
                        f.write(f"      Pages: {analysis['pages']}\n")
                        f.write(f"      Total issues: {sum(analysis['issues'].values())}\n")
                        f.write(f"      Issue #2 problems: {analysis['issues']['issue2_specific']}\n")
                        f.write(f"      Characters: {analysis['stats']['total_chars_raw']:,}\n")
                        
                        if analysis['examples']:
                            f.write(f"      Examples: {analysis['examples'][:2]}\n")
        
        print(f"📋 Comprehensive comparison report saved: {report_file}")
        
        # Print console summary
        print("📊 MULTI-METHOD EXTRACTION SUMMARY:")
        for method, stats in method_stats.items():
            if stats['pdfs_processed'] > 0:
                print(f"   {method.upper()}: {stats['total_issues']} issues, {stats['issue2_problems']} Issue #2 problems")
        
        if ranking:
            best_overall = ranking[0][0]
            print(f"🏆 BEST OVERALL: {best_overall.upper()}")
        
        if issue2_ranking:
            best_issue2 = issue2_ranking[0][0]
            print(f"🎯 BEST FOR ISSUE #2: {best_issue2.upper()}")


def main():
    """Main entry point."""
    tester = MultiPDFTester()
    tester.test_all_pdfs()


if __name__ == "__main__":
    main()