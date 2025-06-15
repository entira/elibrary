#!/usr/bin/env python3
"""
PDF Library Chat Interface with Interactive Library Selection
Interactive chat with PDF library video memory using local Ollama.
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from memvid import MemvidChat

# Cross-platform keyboard input
try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False


class OllamaLLM:
    """Local Ollama LLM interface for chat responses."""
    
    def __init__(self, model: str = "mistral:latest", base_url: str = "http://localhost:11434"):
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
- Put citations at the END of sentences, not in the middle
- Use the exact citation format provided in the context: [Book Title, page X]
- Example: "Balance speed with quality. [Podcasting 100 Success Secrets, page 54]"
- If the context doesn't contain relevant information, say so politely
- Always cite your sources when making claims"""

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 512
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


class PDFLibraryChatV2:
    """Enhanced chat interface with interactive library selection."""
    
    def __init__(self, use_ollama: bool = True):
        self.use_ollama = use_ollama
        
        # Find all available libraries
        available_libraries = self.find_all_libraries()
        
        if not available_libraries:
            raise FileNotFoundError("No video library found. Please run pdf_library_processor.py first!")
        
        # Interactive selection if multiple libraries
        if len(available_libraries) == 1:
            selected_library = available_libraries[0]
            print(f"📚 Using library: {selected_library['name']} ({selected_library['version']})")
        else:
            selected_library = self.select_library_interactive(available_libraries)
        
        self.video_file = selected_library["video_file"]
        self.index_file = selected_library["index_file"]
        self.library_info = selected_library
        
        # Initialize Ollama LLM if requested
        self.llm = OllamaLLM() if use_ollama else None
        
        # Initialize MemvidChat without LLM (we use our own Ollama)
        try:
            self.chat = MemvidChat(str(self.video_file), str(self.index_file), llm_provider=None)
        except:
            # Fallback: initialize without LLM parameters
            from memvid import MemvidRetriever
            self.chat = MemvidRetriever(str(self.video_file), str(self.index_file))
        
        # Session stats
        self.session_stats = {
            "queries": 0,
            "start_time": time.time()
        }
        
        print(f"📚 PDF Library Chat V2 initialized")
        print(f"🎥 Video: {Path(self.video_file).name}")
        print(f"📋 Index: {Path(self.index_file).name}")
        print(f"📊 Version: {self.library_info['version']}")
        print(f"📝 {self.library_info['chunks']} chunks from {self.library_info['files']} files")
        if self.use_ollama:
            print(f"🤖 Using Ollama LLM: {self.llm.model}")
        print()
    
    def find_all_libraries(self) -> List[Dict[str, Any]]:
        """Find all available library files."""
        libraries = []
        possible_dirs = ["./memvid_out"]
        
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                # Check for library files
                patterns = [
                    ("library.mp4", "library_index.json")
                ]
                
                for video_name, index_name in patterns:
                    video_path = os.path.join(dir_path, video_name)
                    index_path = os.path.join(dir_path, index_name)
                    
                    if os.path.exists(video_path) and os.path.exists(index_path):
                        # Get basic info about the library
                        library_info = self.get_library_preview(index_path)
                        
                        libraries.append({
                            "name": f"{Path(dir_path).name}/{video_name}",
                            "video_file": video_path,
                            "index_file": index_path,
                            "directory": dir_path,
                            "chunks": library_info.get("total_chunks", 0),
                            "files": library_info.get("total_files", "Unknown"),
                            "version": "V2 Enhanced" if "v2" in video_name.lower() or "v2" in dir_path.lower() else "V1 Basic",
                            "avg_length": library_info.get("avg_length", "Unknown")
                        })
        
        # Sort by directory name (newest/v2 first)
        libraries.sort(key=lambda x: (x["version"], x["directory"]), reverse=True)
        return libraries
    
    def get_library_preview(self, index_path: str) -> Dict[str, Any]:
        """Get preview information about a library."""
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            info = {
                "total_chunks": len(data.get("metadata", []))
            }
            
            # Check for enhanced stats (V2)
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
    
    def select_library_interactive(self, libraries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Interactive library selection using arrow keys."""
        current_selection = 0
        
        def display_menu():
            os.system('clear' if os.name == 'posix' else 'cls')
            print("📚" + "="*70 + "📚")
            print("🎯 Select PDF Library")
            print("📚" + "="*70 + "📚")
            print()
            print("Enter library number (1-{}) or 'q' to quit:".format(len(libraries)))
            print()
            
            for i, library in enumerate(libraries):
                number = f"[{i+1}]"
                highlight = "→ " if i == current_selection else "  "
                print(f"{highlight}{number} {library['name']}")
                
                print(f"     📂 Directory: {library['directory']}")
                print(f"     📊 Version: {library['version']}")
                print(f"     📝 Chunks: {library['chunks']} (avg: {library['avg_length']})")
                print(f"     📚 Files: {library['files']}")
                print()
        
        # Display initial menu
        display_menu()
        
        # Use simplified input for cross-platform compatibility
        while True:
            try:
                print("📝 Enter your choice: ", end='', flush=True)
                user_input = input().strip().lower()
                
                if user_input == 'q':
                    print("👋 Goodbye!")
                    sys.exit(0)
                elif user_input.isdigit():
                    choice = int(user_input) - 1
                    if 0 <= choice < len(libraries):
                        current_selection = choice
                        break
                    else:
                        print(f"❌ Invalid choice. Please enter 1-{len(libraries)}")
                        display_menu()
                elif user_input == '':
                    # Default to current selection
                    break
                else:
                    print("❌ Invalid input. Enter a number, or 'q' to quit")
                    display_menu()
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                sys.exit(0)
            except EOFError:
                print("\n👋 Goodbye!")
                sys.exit(0)
        
        selected = libraries[current_selection]
        
        # Clear screen and show selection
        os.system('clear' if os.name == 'posix' else 'cls')
        print("✅" + "="*70 + "✅")
        print(f"📚 Selected Library: {selected['name']}")
        print("✅" + "="*70 + "✅")
        print(f"📊 Version: {selected['version']}")
        print(f"📝 Chunks: {selected['chunks']} (avg: {selected['avg_length']})")
        print(f"📚 Files: {selected['files']}")
        print()
        print("Loading library...")
        time.sleep(1.5)
        
        return selected
    
    def load_library_stats(self) -> Dict[str, Any]:
        """Load detailed statistics about the PDF library."""
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            stats = {
                'total_chunks': len(index_data.get('metadata', [])),
                'books': {}
            }
            
            # Check if we have enhanced stats (V2)
            if 'enhanced_stats' in index_data:
                enhanced = index_data['enhanced_stats']
                stats['total_books'] = enhanced.get('total_files', 0)
                stats['cross_page_chunks'] = enhanced.get('cross_page_chunks', 0)
                
                # Get detailed book information
                files_info = enhanced.get('files', {})
                for filename, file_data in files_info.items():
                    clean_name = Path(filename).stem[:50]  # Shorten filename
                    stats['books'][clean_name] = {
                        'title': file_data.get('title', clean_name),
                        'authors': file_data.get('authors', 'Unknown'),
                        'publishers': file_data.get('publishers', 'Unknown'),
                        'year': file_data.get('year', 'Unknown'),
                        'doi': file_data.get('doi', 'Unknown'),
                        'chunks': file_data.get('chunks', 0),
                        'pages': file_data.get('unique_pages', 0)
                    }
            else:
                # Fallback for V1: estimate from PDF directory
                pdf_dir = Path("./pdf_books")
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
        """Display information about the PDF library."""
        stats = self.load_library_stats()
        
        print("📖 Library Overview:")
        print(f"   📚 Total books: {stats['total_books']}")
        print(f"   📝 Total chunks: {stats['total_chunks']}")
        if 'cross_page_chunks' in stats:
            print(f"   🔗 Cross-page chunks: {stats['cross_page_chunks']}")
        print(f"   📊 Version: {self.library_info['version']}")
        print()
        
        print("📑 Books in library:")
        for i, (file_key, info) in enumerate(stats['books'].items(), 1):
            # Display full title without truncation
            title = info['title']
            # Remove list brackets and quotes from authors if present
            authors = str(info['authors']).strip("[]'\"")
            publishers = str(info['publishers']).strip("[]'\"") if info.get('publishers', 'Unknown') != 'Unknown' else 'Unknown'
            doi = info.get('doi', 'Unknown')
            
            print(f"   {i:2d}. {title}")
            print(f"       📖 Author(s): {authors}")
            print(f"       🏢 Publisher(s): {publishers}")
            print(f"       📅 Year: {info['year']}")
            if doi != 'Unknown':
                print(f"       🔗 DOI/ISBN: {doi}")
            if info['pages'] != 'Unknown':
                print(f"       📄 Pages: {info['pages']}")
            print(f"       📝 Chunks: {info['chunks']}")
            print()
    
    def search_library(self, query: str, limit: int = 5) -> str:
        """Search library and return formatted results."""
        try:
            start_time = time.time()
            context_chunks = self.chat.search_context(query, top_k=limit)
            search_time = time.time() - start_time
            
            if not context_chunks:
                return f"🔍 No relevant results found for: '{query}'"
            
            # Format results
            result = f"🔍 Search results for: '{query}' ({search_time:.2f}s)\n\n"
            result += "📄 Relevant passages:\n"
            result += "─" * 60 + "\n"
            
            for i, chunk in enumerate(context_chunks, 1):
                result += f"\n[Result {i}]:\n{chunk}\n"
            
            result += "─" * 60
            return result
            
        except Exception as e:
            return f"❌ Search error: {e}"
    
    def _add_citations_to_context(self, context_chunks: List[str]) -> List[str]:
        """Add source citations to context chunks by matching with index metadata."""
        try:
            # Load index data
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            metadata_list = index_data.get('metadata', [])
            
            # Try to match context chunks with metadata
            context_with_citations = []
            for i, chunk in enumerate(context_chunks):
                # Find matching metadata by text content
                citation = f"[Unknown source, page Unknown]"
                
                # Search for matching chunk in metadata
                for meta in metadata_list:
                    if meta.get('text', '') == chunk.strip():
                        # Found match, add citation info
                        chunk_metadata = meta.get('enhanced_metadata', {})
                        if chunk_metadata:
                            title = chunk_metadata.get('title', 'Unknown title')
                            page_ref = chunk_metadata.get('page_reference', 'Unknown page')
                            citation = f"[{title}, page {page_ref}]"
                        break
                
                # Put citation at the end for cleaner LLM processing
                context_with_citations.append(f"{chunk} {citation}")
            
            return context_with_citations
            
        except Exception as e:
            # Fallback: return context without citations
            return [f"{chunk} [Unknown source]" for i, chunk in enumerate(context_chunks)]
    
    def chat_with_library(self, query: str) -> str:
        """Chat with the library using context and LLM."""
        try:
            start_time = time.time()
            
            # Get context from video memory
            context_chunks = self.chat.search_context(query, top_k=5)
            
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
- Put citations at the END of sentences, not in the middle
- Use the exact citation format provided in the context: [Book Title, page X]
- Example: "Balance speed with quality. [Podcasting 100 Success Secrets, page 54]"
- If the context doesn't contain relevant information, say so politely
- Always cite your sources when making claims"""
                
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
        print("   info          - Show library information")
        print("   search <query>- Search library content")
        print("   stats         - Show session statistics")
        print("   clear         - Clear screen")
        print("   exit/quit     - Exit chat")
        print()
        print("💡 Tips:")
        print("   - Ask questions about the content of your PDF books")
        print("   - Use 'search' to see raw search results")
        print("   - Questions can be about specific topics, authors, or concepts")
        print("   - The system uses semantic search across all PDF content")
    
    def run_chat(self):
        """Run the interactive chat loop."""
        print("🚀 PDF Library Chat V2 started!")
        print("   Type 'help' for commands or ask any question about your books.")
        print("   Type 'exit' or 'quit' to end the session.")
        print()
        
        # Start memvid chat session
        try:
            self.chat.start_session()
        except Exception as e:
            print(f"Warning: Could not start MemvidChat session: {e}")
        
        while True:
            try:
                # Get user input
                user_input = input("🤔 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye! Thanks for using PDF Library Chat V2.")
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
        chat_app = PDFLibraryChatV2(use_ollama=use_ollama)
        chat_app.run_chat()
        
    except Exception as e:
        print(f"❌ Error initializing chat: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()