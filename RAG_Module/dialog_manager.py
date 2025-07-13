"""
Dialog Management System for RAG Module
Manages conversation context, history, and user sessions
"""

import json
import time
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import config as rag_config

logger = logging.getLogger(__name__)

class DialogManager:
    """Manages dialog context and conversation flow"""
    
    def __init__(self):
        self.active_sessions = {}
        self.conversation_histories = {}
        logger.info("DialogManager initialized")
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'created_at': time.time(),
            'last_activity': time.time(),
            'context': {},
            'preferences': {
                'preferred_style': 'professional',
                'language': 'english',
                'detail_level': 'medium'
            }
        }
        
        self.conversation_histories[session_id] = []
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        if session_id in self.active_sessions:
            # Check if session is still valid
            session = self.active_sessions[session_id]
            last_activity = session['last_activity']
            timeout_threshold = time.time() - (rag_config.SESSION_TIMEOUT_MINUTES * 60)
            
            if last_activity > timeout_threshold:
                return session
            else:
                # Session expired
                self.end_session(session_id)
                return None
        
        return None
    
    def update_session_activity(self, session_id: str):
        """Update last activity timestamp for session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['last_activity'] = time.time()
    
    def end_session(self, session_id: str):
        """End a conversation session"""
        if session_id in self.active_sessions:
            # Save conversation history before ending
            self._save_conversation_history(session_id)
            
            del self.active_sessions[session_id]
            if session_id in self.conversation_histories:
                del self.conversation_histories[session_id]
            
            logger.info(f"Ended session: {session_id}")
    
    def add_conversation_turn(self, session_id: str, user_query: str, 
                            system_responses: List[Dict[str, Any]], 
                            selected_response: Optional[int] = None) -> bool:
        """Add a conversation turn to the history"""
        
        if session_id not in self.conversation_histories:
            return False
        
        conversation_turn = {
            'timestamp': time.time(),
            'user_query': user_query,
            'system_responses': system_responses,
            'selected_response': selected_response,
            'turn_id': len(self.conversation_histories[session_id]) + 1
        }
        
        self.conversation_histories[session_id].append(conversation_turn)
        
        # Limit conversation history length
        if len(self.conversation_histories[session_id]) > rag_config.MAX_CONVERSATION_HISTORY:
            self.conversation_histories[session_id] = self.conversation_histories[session_id][-rag_config.MAX_CONVERSATION_HISTORY:]
        
        self.update_session_activity(session_id)
        return True
    
    def get_conversation_history(self, session_id: str, 
                               last_n_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        
        if session_id not in self.conversation_histories:
            return []
        
        history = self.conversation_histories[session_id]
        
        if last_n_turns:
            return history[-last_n_turns:]
        
        return history
    
    def get_conversation_context(self, session_id: str) -> str:
        """Get formatted conversation context for RAG"""
        
        history = self.get_conversation_history(session_id, rag_config.CONTEXT_RETENTION_TURNS)
        
        if not history:
            return ""
        
        context_parts = []
        for turn in history:
            context_parts.append(f"User: {turn['user_query']}")
            
            # Include the selected response or the first one
            selected_idx = turn.get('selected_response', 0)
            if selected_idx is None:
                selected_idx = 0
            if turn['system_responses'] and selected_idx < len(turn['system_responses']):
                response = turn['system_responses'][selected_idx]
                context_parts.append(f"Assistant: {response['content'][:200]}...")
        
        return "\n".join(context_parts)
    
    def extract_query_intent(self, query: str, session_id: str) -> Dict[str, Any]:
        """Extract intent and context from user query"""
        
        intent_info = {
            'query': query,
            'intent_type': 'legal_question',
            'legal_domain': self._identify_legal_domain(query),
            'query_type': self._classify_query_type(query),
            'urgency': self._assess_urgency(query),
            'context_dependent': self._is_context_dependent(query, session_id)
        }
        
        return intent_info
    
    def _identify_legal_domain(self, query: str) -> str:
        """Identify the legal domain of the query"""
        
        query_lower = query.lower()
        
        # Family law keywords
        family_keywords = ['marriage', 'divorce', 'custody', 'alimony', 'adoption', 'family', 'spouse', 'child']
        if any(keyword in query_lower for keyword in family_keywords):
            return 'family'
        
        # Property law keywords
        property_keywords = ['property', 'land', 'ownership', 'title', 'deed', 'real estate', 'boundary']
        if any(keyword in query_lower for keyword in property_keywords):
            return 'property'
        
        # Commercial law keywords
        commercial_keywords = ['contract', 'business', 'commercial', 'company', 'partnership', 'intellectual property']
        if any(keyword in query_lower for keyword in commercial_keywords):
            return 'commercial'
        
        # Labour law keywords
        labour_keywords = ['employment', 'labor', 'worker', 'salary', 'termination', 'workplace']
        if any(keyword in query_lower for keyword in labour_keywords):
            return 'labour'
        
        return 'general'
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of legal query"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'define', 'meaning', 'definition']):
            return 'definition'
        elif any(word in query_lower for word in ['how to', 'procedure', 'process', 'steps']):
            return 'procedure'
        elif any(word in query_lower for word in ['can i', 'am i allowed', 'is it legal', 'rights']):
            return 'rights_inquiry'
        elif any(word in query_lower for word in ['penalty', 'punishment', 'consequences', 'violation']):
            return 'consequences'
        elif any(word in query_lower for word in ['case', 'precedent', 'ruling', 'judgment']):
            return 'case_law'
        else:
            return 'general_inquiry'
    
    def _assess_urgency(self, query: str) -> str:
        """Assess the urgency level of the query"""
        
        query_lower = query.lower()
        
        urgent_keywords = ['urgent', 'emergency', 'immediate', 'asap', 'deadline', 'court date']
        if any(keyword in query_lower for keyword in urgent_keywords):
            return 'high'
        
        moderate_keywords = ['soon', 'quickly', 'time-sensitive', 'pending']
        if any(keyword in query_lower for keyword in moderate_keywords):
            return 'medium'
        
        return 'low'
    
    def _is_context_dependent(self, query: str, session_id: str) -> bool:
        """Check if query depends on previous conversation context"""
        
        context_indicators = ['this', 'that', 'it', 'also', 'additionally', 'furthermore', 'moreover']
        query_lower = query.lower()
        
        # Check for context indicators
        has_indicators = any(indicator in query_lower for indicator in context_indicators)
        
        # Check if there's previous conversation
        has_history = len(self.get_conversation_history(session_id)) > 0
        
        return has_indicators and has_history
    
    def update_user_preferences(self, session_id: str, preferences: Dict[str, str]):
        """Update user preferences for the session"""
        
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['preferences'].update(preferences)
            self.update_session_activity(session_id)
    
    def get_user_preferences(self, session_id: str) -> Dict[str, str]:
        """Get user preferences for the session"""
        
        session = self.get_session(session_id)
        if session:
            return session.get('preferences', {})
        
        return {
            'preferred_style': 'professional',
            'language': 'english',
            'detail_level': 'medium'
        }
    
    def _save_conversation_history(self, session_id: str):
        """Save conversation history to file"""
        
        try:
            if session_id in self.conversation_histories:
                filename = f"{rag_config.CHAT_LOGS_DIR}/conversation_{session_id}_{int(time.time())}.json"
                
                conversation_data = {
                    'session_id': session_id,
                    'session_info': self.active_sessions.get(session_id, {}),
                    'conversation_history': self.conversation_histories[session_id],
                    'saved_at': time.time()
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(conversation_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved conversation history to {filename}")
        
        except Exception as e:
            logger.error(f"Error saving conversation history: {e}")
    
    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session"""
        
        session = self.get_session(session_id)
        history = self.get_conversation_history(session_id)
        
        if not session or not history:
            return {}
        
        # Calculate statistics
        total_turns = len(history)
        session_duration = time.time() - session['created_at']
        
        # Query types distribution
        query_types = {}
        legal_domains = {}
        
        for turn in history:
            # This would require storing intent info, simplified for now
            query = turn['user_query']
            domain = self._identify_legal_domain(query)
            query_type = self._classify_query_type(query)
            
            legal_domains[domain] = legal_domains.get(domain, 0) + 1
            query_types[query_type] = query_types.get(query_type, 0) + 1
        
        return {
            'session_id': session_id,
            'total_turns': total_turns,
            'session_duration_minutes': round(session_duration / 60, 2),
            'legal_domains': legal_domains,
            'query_types': query_types,
            'user_preferences': session.get('preferences', {})
        }
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        
        current_time = time.time()
        timeout_threshold = current_time - (rag_config.SESSION_TIMEOUT_MINUTES * 60)
        
        expired_sessions = []
        for session_id, session in self.active_sessions.items():
            if session['last_activity'] < timeout_threshold:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.end_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
