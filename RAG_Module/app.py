"""
Main Application Runner for RAG Legal Assistant
Provides multiple interfaces: Web UI, CLI, and API
"""

import argparse
import sys
import os
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import subprocess

from rag_system import get_rag_system, RAGSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app for API interface
app = FastAPI(
    title="Sri Lankan Legal AI Assistant API",
    description="RAG-powered legal question answering system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    preferences: Optional[Dict[str, str]] = None
    filters: Optional[Dict[str, str]] = None

class QueryResponse(BaseModel):
    success: bool
    session_id: str
    responses: List[Dict[str, Any]]
    processing_time: float
    error: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    filters: Optional[Dict[str, str]] = None

# Global RAG system instance
rag_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    try:
        rag_system = get_rag_system()
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Sri Lankan Legal AI Assistant API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        status = rag_system.get_system_status()
        return {
            "status": "healthy",
            "system_status": status,
            "timestamp": status.get('timestamp')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a legal query"""
    try:
        result = rag_system.process_query(
            request.query,
            request.session_id,
            request.preferences
        )
        
        return QueryResponse(
            success=result['success'],
            session_id=result['session_id'],
            responses=result.get('responses', []),
            processing_time=result.get('processing_time', 0),
            error=result.get('error')
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_documents(request: SearchRequest):
    """Search legal documents"""
    try:
        results = rag_system.search_legal_documents(
            request.query,
            request.filters,
            request.top_k
        )
        
        return {
            "success": True,
            "query": request.query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information"""
    try:
        session_info = rag_system.get_session_info(session_id)
        return session_info
        
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(session_id: str, turn_id: int, selected_response: int, feedback: Optional[str] = None):
    """Submit feedback for a response"""
    try:
        success = rag_system.update_response_feedback(session_id, turn_id, selected_response, feedback)
        return {"success": success, "message": "Feedback recorded"}
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_web_interface():
    """Run the Streamlit web interface"""
    try:
        print("🚀 Starting Web Interface...")
        print("📱 Opening browser at: http://localhost:8501")
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "frontend.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Web interface stopped")
    except Exception as e:
        print(f"❌ Error running web interface: {e}")

def run_api_server():
    """Run the FastAPI server"""
    try:
        print("🚀 Starting API Server...")
        print("📡 API available at: http://localhost:8000")
        print("📚 API docs at: http://localhost:8000/docs")
        
        uvicorn.run(app, host="localhost", port=8000, log_level="info")
        
    except KeyboardInterrupt:
        print("\n👋 API server stopped")
    except Exception as e:
        print(f"❌ Error running API server: {e}")

def run_cli_interface():
    """Run the command-line interface"""
    try:
        print("=" * 60)
        print("⚖️ Sri Lankan Legal AI Assistant - CLI Mode")
        print("=" * 60)
        
        # Initialize RAG system
        print("🔄 Initializing system...")
        rag_system = get_rag_system()
        
        # Check system status
        status = rag_system.get_system_status()
        if status['overall_status'] == 'operational':
            print("✅ System ready!")
            print(f"📄 {status['vector_database']['total_vectors']:,} legal documents available")
        else:
            print("⚠️ System issues detected")
            return
        
        print("\n💡 Tips:")
        print("  - Ask questions about Sri Lankan law")
        print("  - Type 'help' for commands")
        print("  - Type 'quit' to exit")
        print("  - Type 'status' for system information")
        
        session_id = None
        
        while True:
            try:
                # Get user input
                query = input("\n🔍 Your legal question: ").strip()
                
                if not query:
                    continue
                
                # Handle special commands
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                elif query.lower() == 'help':
                    print("\n📖 Available commands:")
                    print("  help     - Show this help message")
                    print("  status   - Show system status")
                    print("  clear    - Clear conversation history")
                    print("  quit     - Exit the application")
                    continue
                
                elif query.lower() == 'status':
                    status = rag_system.get_system_status()
                    print(f"\n📊 System Status: {status['overall_status']}")
                    print(f"📄 Documents: {status['vector_database']['total_vectors']:,}")
                    print(f"🔧 Sessions: {status['dialog_manager']['active_sessions']}")
                    continue
                
                elif query.lower() == 'clear':
                    session_id = None
                    print("🗑️ Conversation history cleared")
                    continue
                
                # Process query
                print("🤔 Processing your question...")
                
                result = rag_system.process_query(query, session_id)
                
                if not session_id:
                    session_id = result['session_id']
                
                if result['success']:
                    print(f"\n✅ Generated {len(result['responses'])} responses in {result['processing_time']:.2f}s")
                    print(f"📚 Based on {result['retrieved_docs_count']} legal documents")
                    
                    # Show responses
                    for i, response in enumerate(result['responses']):
                        print(f"\n{'='*50}")
                        print(f"📋 Response {i+1}: {response['title']}")
                        if response.get('recommended'):
                            print("⭐ RECOMMENDED")
                        print(f"🎯 Confidence: {response['confidence'].upper()}")
                        print(f"📖 Style: {response['style'].capitalize()}")
                        print(f"{'='*50}")
                        print(response['content'])
                        
                        if response['sources']:
                            print(f"\n📚 Sources ({len(response['sources'])}):")
                            for j, source in enumerate(response['sources'][:3]):
                                print(f"  {j+1}. {source['title']} ({source['type']}) - Score: {source['score']:.3f}")
                    
                    print(f"\n**Legal Disclaimer**: This response is generated by an AI system for informational purposes only. It should not be considered as legal advice. Please consult with a qualified legal professional for specific legal matters.")
                    
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Thank you for using the Legal AI Assistant!")
        
    except Exception as e:
        print(f"❌ CLI interface error: {e}")

def run_tests():
    """Run the test suite"""
    try:
        print("🧪 Running RAG System Tests...")
        
        # Import and run tests
        from test_rag_system import run_comprehensive_tests
        result = run_comprehensive_tests()
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Sri Lankan Legal AI Assistant")
    parser.add_argument(
        "mode",
        choices=["web", "api", "cli", "test"],
        help="Application mode: web (Streamlit UI), api (FastAPI server), cli (command line), test (run tests)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for web interface (default: 8501)"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port for API server (default: 8000)"
    )
    
    args = parser.parse_args()
    
    print("⚖️ Sri Lankan Legal AI Assistant")
    print("🤖 RAG-powered Legal Question Answering System")
    print("=" * 60)
    
    if args.mode == "web":
        run_web_interface()
    
    elif args.mode == "api":
        run_api_server()
    
    elif args.mode == "cli":
        run_cli_interface()
    
    elif args.mode == "test":
        success = run_tests()
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
