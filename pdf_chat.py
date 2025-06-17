#!/usr/bin/env python3
"""
PDF Library Chat Interface with Interactive Library Selection
Interactive chat with PDF library video memory using local Ollama.
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

import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, List

# Import memvid only when needed to avoid dependency issues


class OllamaLLM:
    """Local Ollama LLM interface for chat responses."""
    
    def __init__(self, model: str = "gemma3:4b-it-qat", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using Ollama."""
        try:
            # Combine context and prompt with citation requirements
            full_prompt = f"""Based on the following context from PDF books, answer the user's question:

CONTEXT:
{context}

QUESTION: {prompt}

INSTRUCTIONS:
- Provide a helpful answer based on the context
- IMPORTANT: Use ONLY the exact citations that appear in the context above
- Do NOT create new citations - copy the citation format exactly as shown  
- Put citations at the END of sentences, not in the middle
- If the context doesn't contain relevant information, say so politely
- Each citation is already in the correct format [Book Title, page X - Library Y]"""

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 768
                    },
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            return result.get("response", "Sorry, I couldn't generate a response.")
            
        except Exception as e:
            return f"Error generating response: {e}"


class MultiLibraryRetriever:
    """Multi-library retriever that searches across all available libraries."""
    
    def __init__(self, libraries: List[Dict[str, Any]]):
        self.libraries = libraries
        self.retrievers = {}
        
        # Initialize retrievers for each library
        try:
            # Import memvid only when actually needed
            with suppress_stdout(), suppress_stderr(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from memvid import MemvidRetriever
        except Exception as e:
            print(f"❌ Failed to import MemvidRetriever: {e}")
            print("⚠️  Multi-library search will not be available.")
            print("💡 This might be due to numpy/transformers dependency issues.")
            print("🔄 Try reinstalling dependencies or use a different Python environment.")
            raise ImportError(f"MemvidRetriever import failed: {e}")
            
        for lib in libraries:
            try:
                retriever = MemvidRetriever(lib["video_file"], lib["index_file"])
                self.retrievers[lib["library_id"]] = {
                    "retriever": retriever,
                    "info": lib
                }
                print(f"   ✅ Library {lib['library_id']}: {lib['chunks']} chunks ready")
            except Exception as e:
                print(f"   ❌ Library {lib['library_id']}: Failed to load - {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Search across all libraries and return top results."""
        all_results = []
        
        # Search each library
        for lib_id, lib_data in self.retrievers.items():
            try:
                results = lib_data["retriever"].search(query, top_k=top_k)
                # Add library context to each result
                for result in results:
                    all_results.append({
                        "text": result,
                        "library_id": lib_id,
                        "library_name": lib_data["info"]["name"]
                    })
            except Exception as e:
                print(f"Search error in Library {lib_id}: {e}")
        
        # Sort by relevance (assuming MemvidRetriever returns sorted results)
        # For now, just return the text portions
        return [result["text"] for result in all_results[:top_k]]
    
    def get_library_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all libraries."""
        total_chunks = sum(lib["chunks"] for lib in self.libraries)
        total_files = sum(lib["files"] if isinstance(lib["files"], int) else 0 for lib in self.libraries)
        
        return {
            "total_libraries": len(self.libraries),
            "total_chunks": total_chunks,
            "total_files": total_files,
            "libraries": {lib["library_id"]: lib for lib in self.libraries}
        }


class PDFLibraryChat:
    """Enhanced chat interface with multi-library support."""
    
    def __init__(self, use_ollama: bool = True):
        self.use_ollama = use_ollama
        
        # Find all available libraries
        available_libraries = self.find_all_libraries()
        
        if not available_libraries:
            raise FileNotFoundError("No libraries found. Please run pdf_library_processor.py first!")
        
        print(f"📚 Found {len(available_libraries)} library instances")
        
        # Initialize multi-library retriever
        print("🔄 Initializing multi-library search...")
        self.multi_retriever = MultiLibraryRetriever(available_libraries)
        self.available_libraries = available_libraries
        
        # Initialize Ollama LLM if requested
        self.llm = OllamaLLM() if use_ollama else None
        
        # Session stats
        self.session_stats = {
            "queries": 0,
            "start_time": time.time()
        }
        
        # Display summary
        stats = self.multi_retriever.get_library_stats()
        print(f"📚 Multi-Library Chat initialized")
        print(f"📊 Total libraries: {stats['total_libraries']}")
        print(f"📝 Total chunks: {stats['total_chunks']}")
        print(f"📄 Total files: {stats['total_files']}")
        if self.use_ollama:
            print(f"🤖 Using Ollama LLM: {self.llm.model}")
        print()
    
    def find_all_libraries(self) -> List[Dict[str, Any]]:
        """Find all available library files across multiple library instances."""
        libraries = []
        library_root = Path("./library")
        
        if not library_root.exists():
            return libraries
        
        # Search for numbered library directories
        for item in library_root.iterdir():
            if item.is_dir() and item.name.isdigit():
                data_dir = item / "data"
                
                if data_dir.exists():
                    video_path = data_dir / "library.mp4"
                    index_path = data_dir / "library_index.json"
                    
                    if video_path.exists() and index_path.exists():
                        # Get basic info about the library
                        library_info = self.get_library_preview(str(index_path))
                        
                        libraries.append({
                            "name": f"Library {item.name}",
                            "video_file": str(video_path),
                            "index_file": str(index_path),
                            "directory": str(data_dir),
                            "library_id": item.name,
                            "chunks": library_info.get("total_chunks", 0),
                            "files": library_info.get("total_files", "Unknown"),
                            "version": "Current",
                            "avg_length": library_info.get("avg_length", "Unknown")
                        })
        
        # Sort by library ID (numeric)
        libraries.sort(key=lambda x: int(x["library_id"]))
        return libraries
    
    def get_library_preview(self, index_path: str) -> Dict[str, Any]:
        """Get preview information about a library."""
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            info = {
                "total_chunks": len(data.get("metadata", []))
            }
            
            # Check for enhanced stats
            if "enhanced_stats" in data:
                stats = data["enhanced_stats"]
                info["total_files"] = stats.get("total_files", 0)
            else:
                # Estimate from PDF directory for V1
                pdf_dir = Path("./pdf_books")
                if pdf_dir.exists():
                    info["total_files"] = len(list(pdf_dir.glob("*.pdf")))
                else:
                    info["total_files"] = "Unknown"
            
            # Calculate average length
            metadata = data.get("metadata", [])
            if metadata:
                total_length = sum(item.get("length", 0) for item in metadata)
                info["avg_length"] = f"{total_length // len(metadata)} chars"
            
            return info
        except Exception as e:
            print(f"Error reading {index_path}: {e}")
            return {"total_chunks": 0, "total_files": "Unknown", "avg_length": "Unknown"}
    
    
    def load_library_stats(self, index_file: str = None) -> Dict[str, Any]:
        """Load detailed statistics about the PDF library."""
        # Use provided index_file or fall back to first available library
        if index_file is None and self.available_libraries:
            index_file = self.available_libraries[0]["index_file"]
        elif index_file is None:
            return {'total_books': 0, 'total_chunks': 0, 'books': {}}
            
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            stats = {
                'total_chunks': len(index_data.get('metadata', [])),
                'books': {}
            }
            
            # Check if we have enhanced stats
            if 'enhanced_stats' in index_data:
                enhanced = index_data['enhanced_stats']
                stats['total_books'] = enhanced.get('total_files', 0)
                stats['cross_page_chunks'] = enhanced.get('cross_page_chunks', 0)
                
                # Get detailed book information from metadata
                metadata_list = index_data.get('metadata', [])
                books_by_file = {}
                
                # Group metadata by file to get book-level information
                for meta in metadata_list:
                    if 'enhanced_metadata' in meta:
                        enhanced_meta = meta['enhanced_metadata']
                        file_name = enhanced_meta.get('file_name', '')
                        
                        if file_name not in books_by_file:
                            books_by_file[file_name] = {
                                'title': enhanced_meta.get('title', 'Unknown'),
                                'authors': enhanced_meta.get('authors', ['Unknown']),
                                'publishers': enhanced_meta.get('publishers', ['Unknown']),
                                'year': enhanced_meta.get('year', 'Unknown'),
                                'doi': enhanced_meta.get('doi', 'Unknown'),
                                'chunks': 0,
                                'pages': enhanced_meta.get('num_pages', 0)
                            }
                        books_by_file[file_name]['chunks'] += 1
                
                # Convert to the expected format
                for file_name, book_data in books_by_file.items():
                    clean_name = Path(file_name).stem[:50]
                    stats['books'][clean_name] = book_data
                
                # Update total books count
                stats['total_books'] = len(books_by_file)
            else:
                # Fallback for V1: estimate from PDF directory
                pdf_dir = Path("./library/1/pdf")
                if pdf_dir.exists():
                    pdf_files = list(pdf_dir.glob("*.pdf"))
                    stats['total_books'] = len(pdf_files)
                    
                    for i, pdf_path in enumerate(pdf_files, 1):
                        title = pdf_path.stem.replace("_", " ").replace("-", " ")
                        if len(title) > 50:
                            title = title[:50] + "..."
                        
                        stats['books'][pdf_path.stem] = {
                            'title': title,
                            'authors': 'Unknown',
                            'publishers': 'Unknown',
                            'year': 'Unknown',
                            'doi': 'Unknown',
                            'chunks': stats['total_chunks'] // len(pdf_files),  # Estimate
                            'pages': 'Unknown'
                        }
                else:
                    stats['total_books'] = 0
            
            return stats
            
        except Exception as e:
            print(f"Error loading library stats: {e}")
            return {'total_books': 0, 'total_chunks': 0, 'books': {}}
    
    def show_library_info(self):
        """Display information about all PDF libraries."""
        print("📖 Multi-Library Overview:")
        
        # Get overall stats
        stats = self.multi_retriever.get_library_stats()
        print(f"   📚 Total libraries: {stats['total_libraries']}")
        print(f"   📄 Total files: {stats['total_files']}")
        print(f"   📝 Total chunks: {stats['total_chunks']}")
        print()
        
        # Show individual library details
        for lib_id, lib_info in stats['libraries'].items():
            print(f"📚 Library {lib_id}:")
            detailed_stats = self.load_library_stats(lib_info["index_file"])
            print(f"   📄 Files: {detailed_stats['total_books']}")
            print(f"   📝 Chunks: {detailed_stats['total_chunks']}")
            if 'cross_page_chunks' in detailed_stats:
                print(f"   🔗 Cross-page chunks: {detailed_stats['cross_page_chunks']}")
            print()
        
        # Show books from individual libraries
        print("📑 Books in libraries:")
        book_counter = 1
        for lib_id, lib_info in stats['libraries'].items():
            detailed_stats = self.load_library_stats(lib_info["index_file"])
            if 'books' in detailed_stats:
                print(f"\n📚 Library {lib_id} books:")
                for file_key, info in detailed_stats['books'].items():
                    # Display full title without truncation
                    title = info['title']
                    # Properly format authors and publishers from list format
                    if isinstance(info['authors'], list):
                        authors = ', '.join(info['authors'])
                    else:
                        # Handle string representation of list like "['Author1', 'Author2']"
                        authors_str = str(info['authors'])
                        if authors_str.startswith('[') and authors_str.endswith(']'):
                            # Parse the string representation of list
                            import ast
                            try:
                                authors_list = ast.literal_eval(authors_str)
                                authors = ', '.join(authors_list)
                            except:
                                # Fallback: remove brackets and quotes manually but preserve content
                                authors = authors_str.strip('[]').replace("'", "").replace('"', '')
                        else:
                            authors = authors_str
                    
                    if isinstance(info['publishers'], list):
                        publishers = ', '.join(info['publishers'])
                    else:
                        # Handle string representation of list like "['Publisher1', 'Publisher2']"
                        publishers_str = str(info['publishers'])
                        if publishers_str.startswith('[') and publishers_str.endswith(']'):
                            # Parse the string representation of list
                            import ast
                            try:
                                publishers_list = ast.literal_eval(publishers_str)
                                publishers = ', '.join(publishers_list)
                            except:
                                # Fallback: remove brackets and quotes manually but preserve content
                                publishers = publishers_str.strip('[]').replace("'", "").replace('"', '')
                        else:
                            publishers = publishers_str if publishers_str != 'Unknown' else 'Unknown'
                    doi = info.get('doi', 'Unknown')
                    
                    print(f"   {book_counter:2d}. {title}")
                    print(f"       📖 Author(s): {authors}")
                    print(f"       🏢 Publisher(s): {publishers}")
                    print(f"       📅 Year: {info['year']}")
                    if doi != 'Unknown':
                        print(f"       🔗 DOI/ISBN: {doi}")
                    if info['pages'] != 'Unknown':
                        print(f"       📄 Pages: {info['pages']}")
                    print(f"       📝 Chunks: {info['chunks']}")
                    print()
                    book_counter += 1
    
    def search_library(self, query: str, limit: int = 5) -> str:
        """Search across all libraries and return formatted results with citations."""
        try:
            start_time = time.time()
            context_chunks = self.multi_retriever.search(query, top_k=limit)
            search_time = time.time() - start_time
            
            if not context_chunks:
                return f"🔍 No relevant results found for: '{query}'"
            
            # Add citations to context chunks
            context_with_citations = self._add_citations_to_context(context_chunks)
            
            # Format results
            result = f"🔍 Search results for: '{query}' ({search_time:.2f}s)\n\n"
            result += "📄 Relevant passages:\n"
            result += "─" * 60 + "\n"
            
            for i, chunk_with_citation in enumerate(context_with_citations, 1):
                result += f"\n[Result {i}]:\n{chunk_with_citation}\n"
            
            result += "─" * 60
            return result
            
        except Exception as e:
            return f"❌ Search error: {e}"
    
    def _add_citations_to_context(self, context_chunks: List[str]) -> List[str]:
        """Add source citations to context chunks by matching with multi-library metadata."""
        try:
            # Load all index data from all libraries
            all_metadata = []
            for lib in self.available_libraries:
                try:
                    with open(lib["index_file"], 'r', encoding='utf-8') as f:
                        index_data = json.load(f)
                    metadata_list = index_data.get('metadata', [])
                    # Add library context to each metadata entry
                    for meta in metadata_list:
                        meta['_library_id'] = lib["library_id"]
                        meta['_library_name'] = lib["name"]
                    all_metadata.extend(metadata_list)
                except Exception as e:
                    print(f"Warning: Could not load index for Library {lib['library_id']}: {e}")
                    continue
            
            # Try to match context chunks with metadata
            context_with_citations = []
            for i, chunk in enumerate(context_chunks):
                # Find matching metadata by text content
                citation = f"[Unknown source, page Unknown]"
                
                # Search for matching chunk in all metadata
                for meta in all_metadata:
                    if meta.get('text', '') == chunk.strip():
                        # Found match, add citation info with library context
                        chunk_metadata = meta.get('enhanced_metadata', {})
                        if chunk_metadata:
                            title = chunk_metadata.get('title', 'Unknown title')
                            page_ref = chunk_metadata.get('page_reference', 'Unknown page')
                            library_name = meta.get('_library_name', 'Unknown Library')
                            citation = f"[{title}, page {page_ref} - {library_name}]"
                        break
                
                # Put citation at the end for cleaner LLM processing
                context_with_citations.append(f"{chunk} {citation}")
            
            return context_with_citations
            
        except Exception as e:
            # Fallback: return context without citations
            return [f"{chunk} [Unknown source]" for i, chunk in enumerate(context_chunks)]
    
    def chat_with_library(self, query: str) -> str:
        """Chat with all libraries using context and LLM."""
        try:
            start_time = time.time()
            
            # Get context from multi-library search
            context_chunks = self.multi_retriever.search(query, top_k=5)
            
            if not context_chunks:
                return "🔍 I couldn't find relevant information in the library for your question."
            
            # Load index to get metadata for citations
            context_with_citations = self._add_citations_to_context(context_chunks)
            
            context = "\n\n".join(context_with_citations)
            
            # Generate response using Ollama LLM
            if self.use_ollama and self.llm:
                response = self.llm.generate_response(query, context)
                response_time = time.time() - start_time
                
                # Add debug information with full prompt
                debug_prompt = f"""Based on the following context from PDF books, answer the user's question:

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Provide a helpful answer based on the context
- IMPORTANT: Use ONLY the exact citations that appear in the context above
- Do NOT create new citations - copy the citation format exactly as shown
- Put citations at the END of sentences, not in the middle
- If the context doesn't contain relevant information, say so politely
- Each citation is already in the correct format [Book Title, page X - Library Y]"""
                
                footer = f"\n\n⏱️ Response time: {response_time:.2f}s"
                footer += f"\n\n🔍 DEBUG - Full prompt used:\n" + "="*60 + f"\n{debug_prompt}\n" + "="*60
                return response + footer
            else:
                # Fallback: return context
                response_time = time.time() - start_time
                result = f"📄 Based on the library content:\n\n{context}"
                result += f"\n\n⏱️ Search time: {response_time:.2f}s"
                return result
                
        except Exception as e:
            return f"❌ Chat error: {e}"
    
    def show_session_stats(self):
        """Display session statistics."""
        duration = time.time() - self.session_stats['start_time']
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        
        print(f"📊 Session Statistics:")
        print(f"   ⏱️  Duration: {minutes}m {seconds}s")
        print(f"   💬 Queries: {self.session_stats['queries']}")
        if self.session_stats['queries'] > 0:
            avg_time = duration / self.session_stats['queries']
            print(f"   📈 Avg. query time: {avg_time:.2f}s")
    
    def show_help(self):
        """Display help information."""
        print("🆘 Available Commands:")
        print("   help          - Show this help message")
        print("   info          - Show multi-library information")
        print("   search <query>- Search across all libraries")
        print("   stats         - Show session statistics")
        print("   clear         - Clear screen")
        print("   exit/quit     - Exit chat")
        print()
        print("💡 Tips:")
        print("   - Ask questions about content from all your PDF libraries")
        print("   - Use 'search' to see raw search results from all libraries")
        print("   - Questions can be about specific topics, authors, or concepts")
        print("   - The system searches across ALL library instances")
        print("   - Citations include library source and PDF page numbers")
        print("   - Results are ranked by relevance across all libraries")
    
    def run_chat(self):
        """Run the interactive chat loop."""
        print("🚀 Multi-Library Chat started!")
        print("   Type 'help' for commands or ask any question about your libraries.")
        print("   Type 'exit' or 'quit' to end the session.")
        print()
        
        # MemvidRetriever doesn't need session start
        # Ready to process queries
        
        while True:
            try:
                # Get user input
                user_input = input("🤔 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye! Thanks for using PDF Library Chat.")
                    break
                
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                elif user_input.lower() == 'info':
                    self.show_library_info()
                    continue
                
                elif user_input.lower() == 'stats':
                    self.show_session_stats()
                    continue
                
                elif user_input.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                
                elif user_input.lower().startswith('search '):
                    query = user_input[7:].strip()
                    if query:
                        print(f"🤖 {self.search_library(query)}")
                    else:
                        print("❌ Please provide a search query. Usage: search <your query>")
                    continue
                
                # Regular chat
                self.session_stats['queries'] += 1
                print("🤖 Assistant:", end=" ")
                response = self.chat_with_library(user_input)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'help' for assistance.\n")


def main():
    """Main entry point."""
    # Test Ollama connection
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        ollama_available = response.status_code == 200
    except:
        ollama_available = False
    
    if not ollama_available:
        print("⚠️  Warning: Ollama not available at localhost:11434")
        print("   Chat will use basic context search without LLM responses.")
        print("   Start Ollama server for enhanced chat experience.")
        use_ollama = False
    else:
        use_ollama = True
    
    # Initialize and run chat
    try:
        chat_app = PDFLibraryChat(use_ollama=use_ollama)
        chat_app.run_chat()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n🔧 Possible solutions:")
        print("   1. Check if memvid is properly installed: pip install memvid")
        print("   2. Try reinstalling numpy and transformers:")
        print("      pip uninstall numpy transformers -y")
        print("      pip install numpy transformers")
        print("   3. Use a clean Python environment or conda environment")
        print("   4. Check if your Python environment is compatible")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        print("\n💡 To create libraries:")
        print("   1. mkdir -p library/1/pdf")
        print("   2. cp your_pdfs/*.pdf library/1/pdf/")
        print("   3. python3 pdf_library_processor.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        print("\n🔍 Please check your setup and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
