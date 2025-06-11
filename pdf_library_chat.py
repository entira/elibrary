#!/usr/bin/env python3
"""
PDF Library Chat Interface
Interactive chat with PDF library video memory using local Ollama.
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from memvid import MemvidChat


class OllamaLLM:
    """Local Ollama LLM interface for chat responses."""
    
    def __init__(self, model: str = "mistral:latest", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using Ollama."""
        try:
            # Combine context and prompt
            full_prompt = f"""Based on the following context from PDF books, answer the user's question:

CONTEXT:
{context}

QUESTION: {prompt}

Please provide a helpful answer based on the context. If the context doesn't contain relevant information, say so politely."""

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


class PDFLibraryChat:
    """Chat interface for PDF library video memory."""
    
    def __init__(self, 
                 video_file: str = "./memvid_out_2/library.mp4",
                 index_file: str = "./memvid_out_2/library_index.json",
                 use_ollama: bool = True):
        
        self.video_file = Path(video_file)
        self.index_file = Path(index_file)
        self.use_ollama = use_ollama
        
        # Validate files exist
        if not self.video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")
        if not self.index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        # Initialize Ollama LLM if requested
        self.llm = OllamaLLM() if use_ollama else None
        
        # Initialize MemvidChat (without API key for local use)
        self.chat = MemvidChat(str(self.video_file), str(self.index_file))
        
        # Session stats
        self.session_stats = {
            "queries": 0,
            "start_time": time.time()
        }
        
        print(f"📚 PDF Library Chat initialized")
        print(f"🎥 Video: {self.video_file.name}")
        print(f"📋 Index: {self.index_file.name}")
        if self.use_ollama:
            print(f"🤖 Using Ollama LLM: {self.llm.model}")
        print()
    
    def load_library_stats(self) -> Dict[str, Any]:
        """Load statistics about the PDF library."""
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # Get total chunks from metadata
            total_chunks = len(index_data.get('metadata', []))
            
            # Count PDF files in the source directory (since memvid doesn't preserve individual book info)
            pdf_dir = Path("./pdf_books")
            if pdf_dir.exists():
                pdf_files = list(pdf_dir.glob("*.pdf"))
                books = {}
                for i, pdf_path in enumerate(pdf_files, 1):
                    # Extract title from filename (remove extension and clean up)
                    title = pdf_path.stem.replace("_", " ").replace("-", " ")
                    # Limit title length
                    if len(title) > 100:
                        title = title[:100] + "..."
                    
                    books[pdf_path.name] = {
                        'title': title,
                        'authors': 'Unknown',
                        'year': 'Unknown', 
                        'chunks': total_chunks // len(pdf_files)  # Estimate chunks per book
                    }
                
                return {
                    'total_books': len(pdf_files),
                    'total_chunks': total_chunks,
                    'books': books
                }
            else:
                return {
                    'total_books': 0,
                    'total_chunks': total_chunks,
                    'books': {}
                }
            
        except Exception as e:
            print(f"Error loading library stats: {e}")
            return {'total_books': 0, 'total_chunks': 0, 'books': {}}
    
    def show_library_info(self):
        """Display information about the PDF library."""
        stats = self.load_library_stats()
        
        print("📖 Library Overview:")
        print(f"   📚 Total books: {stats['total_books']}")
        print(f"   📝 Total chunks: {stats['total_chunks']}")
        print()
        
        print("📑 Books in library:")
        for i, (file_name, info) in enumerate(stats['books'].items(), 1):
            title = info['title'][:50] + '...' if len(info['title']) > 50 else info['title']
            authors = info['authors'][:30] + '...' if len(info['authors']) > 30 else info['authors']
            print(f"   {i:2d}. {title}")
            print(f"       📖 Author(s): {authors}")
            print(f"       📅 Year: {info['year']}")
            print(f"       📝 Chunks: {info['chunks']}")
            print()
    
    def search_library(self, query: str, limit: int = 5) -> str:
        """Search library and return formatted results."""
        try:
            # Use MemvidChat's search functionality
            start_time = time.time()
            context_chunks = self.chat.search_context(query, top_k=limit)
            search_time = time.time() - start_time
            
            if not context_chunks:
                return f"🔍 No relevant results found for: '{query}'"
            
            # Format results
            result = f"🔍 Search results for: '{query}' ({search_time:.2f}s)\n\n"
            result += "📄 Relevant passages:\n"
            result += "─" * 50 + "\n"
            
            # Join the context chunks with separators
            for i, chunk in enumerate(context_chunks, 1):
                result += f"\n[Result {i}]:\n{chunk}\n"
            
            result += "─" * 50
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Search error: {e}"
    
    def chat_with_library(self, query: str) -> str:
        """Chat with the library using context and LLM."""
        try:
            start_time = time.time()
            
            # Get context from video memory
            context_chunks = self.chat.search_context(query, top_k=5)
            
            if not context_chunks:
                return "🔍 I couldn't find relevant information in the library for your question."
            
            # Join context chunks into a single string for the LLM
            context = "\n\n".join([f"[Context {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])
            
            # Generate response using Ollama LLM
            if self.use_ollama and self.llm:
                response = self.llm.generate_response(query, context)
                response_time = time.time() - start_time
                
                # Add metadata about response
                footer = f"\n\n⏱️ Response time: {response_time:.2f}s"
                return response + footer
            else:
                # Fallback: just return context
                response_time = time.time() - start_time
                result = f"📄 Based on the library content:\n\n{context}"
                result += f"\n\n⏱️ Search time: {response_time:.2f}s"
                return result
                
        except Exception as e:
            import traceback
            traceback.print_exc()
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
        print("🚀 PDF Library Chat started!")
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
    # Check if video and index files exist
    video_file = "./memvid_out_2/library.mp4"
    index_file = "./memvid_out_2/library_index.json"
    
    if not Path(video_file).exists():
        print("❌ Error: Video file not found!")
        print(f"   Expected: {video_file}")
        print("   Please run pdf_library_processor.py first to create the video index.")
        sys.exit(1)
    
    if not Path(index_file).exists():
        print("❌ Error: Index file not found!")
        print(f"   Expected: {index_file}")
        print("   Please run pdf_library_processor.py first to create the index.")
        sys.exit(1)
    
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
        chat_app = PDFLibraryChat(
            video_file=video_file,
            index_file=index_file,
            use_ollama=use_ollama
        )
        chat_app.run_chat()
        
    except Exception as e:
        print(f"❌ Error initializing chat: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()