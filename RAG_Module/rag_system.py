"""
Main RAG System Integration
Combines vector retrieval, response generation, and dialog management
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from vector_db_connector import VectorDBConnector
from response_generator import ResponseGenerator
from dialog_manager import DialogManager
import config as rag_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGSystem:
    """Complete RAG System for Legal Question Answering"""
    
    def __init__(self):
        """Initialize the RAG system components"""
        try:
            self.vector_db = VectorDBConnector()
            self.response_generator = ResponseGenerator()
            self.dialog_manager = DialogManager()
            
            logger.info("RAG System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise
    
    def process_query(self, query: str, session_id: Optional[str] = None,
                     user_preferences: Optional[Dict[str, str]] = None,
                     filters: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process a user query and generate responses"""
        
        start_time = time.time()
        
        try:
            # Create session if not provided
            if not session_id:
                session_id = self.dialog_manager.create_session()
            
            # Update user preferences if provided
            if user_preferences:
                self.dialog_manager.update_user_preferences(session_id, user_preferences)

            # Check if this is a simple greeting or non-legal query
            simple_response = self._check_for_simple_response(query)
            if simple_response:
                # Generate simple responses without vector DB search
                responses = self._generate_simple_responses(query, simple_response)

                # Add conversation turn to history
                self.dialog_manager.add_conversation_turn(session_id, query, responses)

                processing_time = time.time() - start_time
                return {
                    'session_id': session_id,
                    'query': query,
                    'enhanced_query': query,
                    'intent_info': {'query_type': 'simple_interaction'},
                    'responses': responses,
                    'retrieved_docs_count': 0,
                    'processing_time': round(processing_time, 3),
                    'timestamp': time.time(),
                    'success': True
                }

            # Extract query intent and context
            intent_info = self.dialog_manager.extract_query_intent(query, session_id)
            
            # Get conversation context
            conversation_context = self.dialog_manager.get_conversation_context(session_id)
            
            # Enhance query with context if needed
            enhanced_query = self._enhance_query_with_context(query, conversation_context, intent_info)
            
            # Retrieve relevant documents
            retrieved_docs = self._retrieve_documents(enhanced_query, intent_info, filters)
            
            # Generate multiple response variants
            responses = self.response_generator.generate_responses(
                query, retrieved_docs, 
                self.dialog_manager.get_conversation_history(session_id, 3)
            )
            
            # Rank responses based on user preferences
            ranked_responses = self._rank_responses(responses, session_id)
            
            # Add conversation turn to history
            self.dialog_manager.add_conversation_turn(session_id, query, ranked_responses)
            
            # Prepare final result
            processing_time = time.time() - start_time
            
            result = {
                'session_id': session_id,
                'query': query,
                'enhanced_query': enhanced_query,
                'intent_info': intent_info,
                'responses': ranked_responses,
                'retrieved_docs_count': len(retrieved_docs),
                'processing_time': round(processing_time, 3),
                'timestamp': time.time(),
                'success': True
            }
            
            logger.info(f"Successfully processed query in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'session_id': session_id,
                'query': query,
                'error': str(e),
                'success': False,
                'timestamp': time.time()
            }
    
    def _enhance_query_with_context(self, query: str, context: str, 
                                   intent_info: Dict[str, Any]) -> str:
        """Enhance query with conversation context"""
        
        if not context or not intent_info.get('context_dependent'):
            return query
        
        # Simple context enhancement
        enhanced_query = f"Context: {context}\n\nCurrent question: {query}"
        return enhanced_query
    
    def _retrieve_documents(self, query: str, intent_info: Dict[str, Any],
                           filters: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using hybrid search"""
        
        try:
            # Determine search filters based on intent and user filters
            search_filters = {}

            # Add intent-based filters
            legal_domain = intent_info.get('legal_domain')
            if legal_domain and legal_domain != 'general':
                search_filters['primary_category'] = legal_domain.capitalize()

            # Add user-provided filters (these override intent-based filters)
            if filters:
                search_filters.update(filters)

            # Perform hybrid search
            if search_filters:
                retrieved_docs = self.vector_db.search_documents(
                    query,
                    top_k=rag_config.RETRIEVAL_TOP_K,
                    filters=search_filters
                )
            else:
                retrieved_docs = self.vector_db.hybrid_search(
                    query,
                    top_k=rag_config.RETRIEVAL_TOP_K
                )
            
            # Filter by confidence threshold
            filtered_docs = self.vector_db.filter_by_confidence(
                retrieved_docs, 
                rag_config.MIN_SIMILARITY_THRESHOLD
            )
            
            # Re-rank results
            reranked_docs = self.vector_db.rerank_results(filtered_docs, query)
            
            # Limit to top results
            final_docs = reranked_docs[:rag_config.RERANK_TOP_K]
            
            logger.info(f"Retrieved {len(final_docs)} relevant documents")
            return final_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _rank_responses(self, responses: List[Dict[str, Any]], 
                       session_id: str) -> List[Dict[str, Any]]:
        """Rank responses based on user preferences and quality"""
        
        try:
            user_prefs = self.dialog_manager.get_user_preferences(session_id)
            preferred_style = user_prefs.get('preferred_style', 'professional')
            
            # Score responses
            for response in responses:
                score = 0.0
                
                # Style preference bonus
                if response['style'] == preferred_style:
                    score += 0.3
                
                # Confidence bonus
                confidence = response['confidence']
                if confidence == 'high':
                    score += 0.3
                elif confidence == 'medium':
                    score += 0.2
                else:
                    score += 0.1
                
                # Source count bonus
                num_sources = response.get('num_sources', 0)
                if num_sources is not None:
                    score += min(num_sources * 0.1, 0.3)
                
                # Content length appropriateness
                content = response.get('content', '')
                if content:
                    content_length = len(content)
                    if 200 <= content_length <= 800:
                        score += 0.1
                
                response['ranking_score'] = score
            
            # Sort by ranking score
            ranked_responses = sorted(responses, key=lambda x: x['ranking_score'], reverse=True)
            
            # Add ranking information
            for i, response in enumerate(ranked_responses):
                response['rank'] = i + 1
                response['recommended'] = (i == 0)
            
            return ranked_responses
            
        except Exception as e:
            logger.error(f"Error ranking responses: {e}")
            return responses
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session information"""
        
        session = self.dialog_manager.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        history = self.dialog_manager.get_conversation_history(session_id)
        stats = self.dialog_manager.get_session_statistics(session_id)
        
        return {
            'session': session,
            'conversation_history': history,
            'statistics': stats,
            'preferences': self.dialog_manager.get_user_preferences(session_id)
        }
    
    def update_response_feedback(self, session_id: str, turn_id: int, 
                               selected_response: int, feedback: Optional[str] = None):
        """Update feedback for a specific response"""
        
        try:
            history = self.dialog_manager.get_conversation_history(session_id)
            
            if turn_id <= len(history):
                turn = history[turn_id - 1]
                turn['selected_response'] = selected_response
                turn['user_feedback'] = feedback
                turn['feedback_timestamp'] = time.time()
                
                logger.info(f"Updated feedback for session {session_id}, turn {turn_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating feedback: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and statistics"""
        
        try:
            # Vector database stats
            db_stats = self.vector_db.get_database_stats()
            
            # Active sessions
            active_sessions = len(self.dialog_manager.active_sessions)
            
            # System health
            system_status = {
                'vector_database': {
                    'status': 'operational' if db_stats.get('total_vectors', 0) > 0 else 'no_data',
                    'total_vectors': db_stats.get('total_vectors', 0),
                    'dimension': db_stats.get('dimension', 0)
                },
                'response_generator': {
                    'status': 'operational',
                    'llm_available': self.response_generator.llm_available
                },
                'dialog_manager': {
                    'status': 'operational',
                    'active_sessions': active_sessions
                },
                'overall_status': 'operational',
                'timestamp': time.time()
            }
            
            return system_status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'overall_status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def cleanup_system(self):
        """Perform system cleanup tasks"""
        
        try:
            # Clean up expired sessions
            self.dialog_manager.cleanup_expired_sessions()
            
            logger.info("System cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during system cleanup: {e}")
    
    def search_legal_documents(self, query: str, filters: Optional[Dict[str, str]] = None,
                             top_k: int = 10) -> List[Dict[str, Any]]:
        """Direct document search interface"""
        
        try:
            if filters:
                results = self.vector_db.search_documents(query, top_k, filters)
            else:
                results = self.vector_db.hybrid_search(query, top_k)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_document_context(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get context around a specific document"""
        
        try:
            return self.vector_db.get_document_context(doc_id)
            
        except Exception as e:
            logger.error(f"Error getting document context: {e}")
            return []

    def _check_for_simple_response(self, query: str) -> Optional[str]:
        """Check if query is a simple greeting or non-legal question"""

        query_lower = query.lower().strip()
        logger.info(f"Checking for simple response: '{query_lower}'")

        # First check if it contains legal keywords - if so, it's NOT a simple response
        legal_keywords = ['law', 'legal', 'court', 'property', 'divorce', 'marriage', 'contract',
                         'employment', 'rights', 'act', 'section', 'clause', 'procedure',
                         'sri lanka', 'sri lankan', 'custody', 'alimony', 'ownership', 'title']
        if any(keyword in query_lower for keyword in legal_keywords):
            logger.info(f"Contains legal keywords, not a simple response: {query_lower}")
            return None

        # Exact match greetings (single words or short phrases)
        if query_lower in ['hi', 'hello', 'hey', 'yo', 'sup', 'hiya', 'howdy']:
            logger.info(f"Detected greeting: {query_lower}")
            return 'greeting'

        # Exact match time-based greetings
        if query_lower in ['good morning', 'good afternoon', 'good evening', 'good night']:
            logger.info(f"Detected greeting: {query_lower}")
            return 'greeting'

        # What's up variations (exact matches)
        if query_lower in ["what's up", "whats up", "what's up?", "whats up?"]:
            logger.info(f"Detected greeting: {query_lower}")
            return 'greeting'

        # How are you variations (exact matches)
        how_are_you_exact = ['how are you', 'how are you?', 'how do you do', 'how are things',
                            'how you doing', 'how are u', 'how r u', 'how are ya']
        if query_lower in how_are_you_exact:
            logger.info(f"Detected how are you: {query_lower}")
            return 'how_are_you'

        # Help requests (exact matches)
        help_requests_exact = ['how can you help', 'what can you do', 'help me', 'can you help',
                              'what do you do', 'what are you', 'who are you', 'what is this']
        if query_lower in help_requests_exact:
            logger.info(f"Detected help request: {query_lower}")
            return 'help_request'

        # Thank you (exact matches)
        thanks_exact = ['thank you', 'thanks', 'thank u', 'thx', 'ty', 'cheers']
        if query_lower in thanks_exact:
            logger.info(f"Detected thanks: {query_lower}")
            return 'thanks'

        # Goodbye (exact matches)
        goodbyes_exact = ['bye', 'goodbye', 'see you', 'farewell', 'cya', 'see ya', 'later']
        if query_lower in goodbyes_exact:
            logger.info(f"Detected goodbye: {query_lower}")
            return 'goodbye'

        logger.info(f"No simple response pattern found for: {query_lower}")
        return None

    def _generate_simple_responses(self, query: str, response_type: str) -> List[Dict[str, Any]]:
        """Generate simple responses for non-legal queries"""

        responses_map = {
            'greeting': {
                'professional': "Hello! I'm your Sri Lankan Legal AI Assistant. I'm here to help you with questions about Sri Lankan law. How may I assist you today?",
                'detailed': "Greetings! Welcome to the Sri Lankan Legal AI Assistant. I specialize in providing information about Sri Lankan legal matters including family law, property law, commercial law, and employment law. Please feel free to ask me any legal questions you may have.",
                'concise': "Hello! I'm here to help with Sri Lankan legal questions. What would you like to know?"
            },
            'how_are_you': {
                'professional': "Thank you for asking. I'm functioning well and ready to assist you with your Sri Lankan legal inquiries. How may I help you today?",
                'detailed': "I'm doing well, thank you! As an AI legal assistant, I'm always ready to help with questions about Sri Lankan law. I can provide information on various legal topics including property rights, family law, employment matters, and commercial regulations.",
                'concise': "I'm doing well! Ready to help with your legal questions."
            },
            'help_request': {
                'professional': "I can assist you with various aspects of Sri Lankan law including: property ownership rights, family law matters, employment regulations, commercial law, and legal procedures. Please ask me specific legal questions and I'll provide detailed information based on Sri Lankan legal documents.",
                'detailed': "I'm here to help with Sri Lankan legal matters! I can provide information on:\n\n• Property and land law\n• Family law (marriage, divorce, custody)\n• Employment and labor law\n• Commercial and business law\n• Legal procedures and rights\n\nSimply ask me any specific legal question, and I'll search through Sri Lankan legal documents to provide you with accurate information.",
                'concise': "I help with Sri Lankan legal questions on property, family, employment, and commercial law. What do you need to know?"
            },
            'thanks': {
                'professional': "You're welcome! I'm glad I could assist you with your legal inquiry. If you have any other questions about Sri Lankan law, please don't hesitate to ask.",
                'detailed': "You're very welcome! I'm pleased to have been able to help you with your legal question. Remember, while I provide information based on Sri Lankan legal documents, for specific legal advice tailored to your situation, it's always best to consult with a qualified legal professional.",
                'concise': "You're welcome! Feel free to ask if you have more legal questions."
            },
            'goodbye': {
                'professional': "Thank you for using the Sri Lankan Legal AI Assistant. Have a good day, and please return if you need any legal information in the future.",
                'detailed': "Goodbye! Thank you for using our legal assistance service. Remember that while I provide information based on Sri Lankan legal documents, for specific legal matters, consulting with a qualified lawyer is always recommended. Have a wonderful day!",
                'concise': "Goodbye! Come back anytime for legal questions."
            }
        }

        response_texts = responses_map.get(response_type, responses_map['greeting'])

        responses = []
        styles = ['professional', 'detailed', 'concise']
        titles = ['Professional Response', 'Detailed Explanation', 'Concise Answer']

        for i, style in enumerate(styles):
            responses.append({
                'style': style,
                'title': titles[i],
                'content': response_texts[style],
                'confidence': 'high',
                'sources': [],
                'timestamp': time.time(),
                'query': query,
                'num_sources': 0,
                'disclaimer': "This is a general response. For specific legal advice, please consult a qualified legal professional.",
                'ranking_score': 0.8 if style == 'professional' else 0.7,
                'rank': i + 1,
                'recommended': (i == 0)
            })

        return responses

# Global RAG system instance
rag_system = None

def get_rag_system() -> RAGSystem:
    """Get or create the global RAG system instance"""
    global rag_system
    
    if rag_system is None:
        rag_system = RAGSystem()
    
    return rag_system

def initialize_rag_system() -> RAGSystem:
    """Initialize the RAG system"""
    global rag_system
    rag_system = RAGSystem()
    return rag_system

if __name__ == "__main__":
    # Test the RAG system
    print("=== RAG System Test ===")
    
    try:
        # Initialize system
        system = RAGSystem()
        
        # Test system status
        status = system.get_system_status()
        print(f"System Status: {status['overall_status']}")
        print(f"Vector Database: {status['vector_database']['total_vectors']} vectors")
        
        # Test query processing
        test_query = "What are the property ownership rights in Sri Lanka?"
        print(f"\nTesting query: {test_query}")
        
        result = system.process_query(test_query)
        
        if result['success']:
            print(f"✅ Query processed successfully in {result['processing_time']}s")
            print(f"📄 Retrieved {result['retrieved_docs_count']} documents")
            print(f"💬 Generated {len(result['responses'])} response variants")
            
            # Show response titles
            for i, response in enumerate(result['responses']):
                print(f"   {i+1}. {response['title']} (Confidence: {response['confidence']})")
        else:
            print(f"❌ Query processing failed: {result.get('error')}")
        
        print("\n🎉 RAG System test completed!")
        
    except Exception as e:
        print(f"❌ RAG System test failed: {e}")
