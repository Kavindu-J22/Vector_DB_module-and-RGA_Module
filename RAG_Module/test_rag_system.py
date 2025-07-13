"""
Comprehensive Test Suite for RAG System
Tests all components and functionality
"""

import unittest
import time
import logging
from typing import Dict, Any, List
from rag_system import RAGSystem
from vector_db_connector import VectorDBConnector
from response_generator import ResponseGenerator
from dialog_manager import DialogManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestVectorDBConnector(unittest.TestCase):
    """Test Vector Database Connector"""
    
    def setUp(self):
        self.connector = VectorDBConnector()
    
    def test_connection(self):
        """Test database connection"""
        stats = self.connector.get_database_stats()
        self.assertIsInstance(stats, dict)
        self.assertGreater(stats.get('total_vectors', 0), 0)
    
    def test_query_embedding(self):
        """Test query embedding generation"""
        query = "property ownership rights"
        embedding = self.connector.create_query_embedding(query)
        
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 384)
        self.assertTrue(all(isinstance(x, float) for x in embedding))
    
    def test_document_search(self):
        """Test document search functionality"""
        query = "marriage and divorce laws"
        results = self.connector.search_documents(query, top_k=5)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 5)
        
        if results:
            result = results[0]
            self.assertIn('id', result)
            self.assertIn('score', result)
            self.assertIn('content', result)
            self.assertIn('metadata', result)
    
    def test_hybrid_search(self):
        """Test hybrid search functionality"""
        query = "commercial contract disputes"
        results = self.connector.hybrid_search(query, top_k=3)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)
        
        if results:
            result = results[0]
            self.assertIn('hybrid_score', result)
            self.assertIn('keyword_score', result)
    
    def test_confidence_filtering(self):
        """Test confidence-based filtering"""
        query = "employment termination"
        results = self.connector.search_documents(query, top_k=10)
        
        if results:
            filtered_results = self.connector.filter_by_confidence(results, 0.3)
            self.assertLessEqual(len(filtered_results), len(results))

class TestResponseGenerator(unittest.TestCase):
    """Test Response Generator"""
    
    def setUp(self):
        self.generator = ResponseGenerator()
    
    def test_initialization(self):
        """Test response generator initialization"""
        self.assertIsInstance(self.generator, ResponseGenerator)
        self.assertIsInstance(self.generator.llm_available, bool)
    
    def test_response_generation(self):
        """Test response generation with mock documents"""
        query = "What are property rights in Sri Lanka?"
        
        mock_docs = [
            {
                'id': 'test_doc_001',
                'score': 0.85,
                'content': 'Property rights in Sri Lanka are governed by various legal provisions...',
                'metadata': {
                    'document_type': 'act',
                    'category': 'Property',
                    'filename': 'Property Rights Act'
                }
            }
        ]
        
        responses = self.generator.generate_responses(query, mock_docs)
        
        self.assertIsInstance(responses, list)
        self.assertEqual(len(responses), 3)  # Should generate 3 variants
        
        for response in responses:
            self.assertIn('style', response)
            self.assertIn('title', response)
            self.assertIn('content', response)
            self.assertIn('confidence', response)
            self.assertIn('sources', response)
            self.assertIn('timestamp', response)
    
    def test_no_results_response(self):
        """Test response generation when no documents found"""
        query = "completely unrelated query about space travel"
        responses = self.generator.generate_responses(query, [])
        
        self.assertIsInstance(responses, list)
        self.assertEqual(len(responses), 3)
        
        for response in responses:
            self.assertEqual(response['confidence'], 'low')
            self.assertEqual(len(response['sources']), 0)
    
    def test_confidence_assessment(self):
        """Test confidence level assessment"""
        # High confidence docs
        high_conf_docs = [{'score': 0.9}, {'score': 0.85}]
        confidence = self.generator._assess_confidence(high_conf_docs)
        self.assertEqual(confidence, 'high')
        
        # Low confidence docs
        low_conf_docs = [{'score': 0.3}, {'score': 0.2}]
        confidence = self.generator._assess_confidence(low_conf_docs)
        self.assertEqual(confidence, 'low')

class TestDialogManager(unittest.TestCase):
    """Test Dialog Manager"""
    
    def setUp(self):
        self.dialog_manager = DialogManager()
    
    def test_session_creation(self):
        """Test session creation and management"""
        session_id = self.dialog_manager.create_session()
        
        self.assertIsInstance(session_id, str)
        self.assertIn(session_id, self.dialog_manager.active_sessions)
        
        session = self.dialog_manager.get_session(session_id)
        self.assertIsNotNone(session)
        self.assertIn('created_at', session)
        self.assertIn('preferences', session)
    
    def test_conversation_history(self):
        """Test conversation history management"""
        session_id = self.dialog_manager.create_session()
        
        # Add conversation turn
        query = "Test legal question"
        responses = [{'content': 'Test response', 'style': 'professional'}]
        
        success = self.dialog_manager.add_conversation_turn(session_id, query, responses)
        self.assertTrue(success)
        
        # Get history
        history = self.dialog_manager.get_conversation_history(session_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['user_query'], query)
    
    def test_intent_extraction(self):
        """Test query intent extraction"""
        session_id = self.dialog_manager.create_session()
        
        # Test different query types
        test_queries = [
            ("What is property ownership?", "definition"),
            ("How to file for divorce?", "procedure"),
            ("Can I terminate employment?", "rights_inquiry"),
            ("Property rights in marriage", "property")
        ]
        
        for query, expected in test_queries:
            intent = self.dialog_manager.extract_query_intent(query, session_id)
            
            self.assertIsInstance(intent, dict)
            self.assertIn('query', intent)
            self.assertIn('legal_domain', intent)
            self.assertIn('query_type', intent)
    
    def test_user_preferences(self):
        """Test user preferences management"""
        session_id = self.dialog_manager.create_session()
        
        # Update preferences
        new_prefs = {'preferred_style': 'detailed', 'language': 'sinhala'}
        self.dialog_manager.update_user_preferences(session_id, new_prefs)
        
        # Get preferences
        prefs = self.dialog_manager.get_user_preferences(session_id)
        self.assertEqual(prefs['preferred_style'], 'detailed')
        self.assertEqual(prefs['language'], 'sinhala')

class TestRAGSystem(unittest.TestCase):
    """Test Complete RAG System"""
    
    def setUp(self):
        self.rag_system = RAGSystem()
    
    def test_system_initialization(self):
        """Test RAG system initialization"""
        self.assertIsInstance(self.rag_system.vector_db, VectorDBConnector)
        self.assertIsInstance(self.rag_system.response_generator, ResponseGenerator)
        self.assertIsInstance(self.rag_system.dialog_manager, DialogManager)
    
    def test_system_status(self):
        """Test system status reporting"""
        status = self.rag_system.get_system_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('overall_status', status)
        self.assertIn('vector_database', status)
        self.assertIn('response_generator', status)
        self.assertIn('dialog_manager', status)
    
    def test_query_processing(self):
        """Test end-to-end query processing"""
        query = "What are the marriage laws in Sri Lanka?"
        
        result = self.rag_system.process_query(query)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('session_id', result)
        self.assertIn('responses', result)
        
        if result['success']:
            self.assertIsInstance(result['responses'], list)
            self.assertEqual(len(result['responses']), 3)
            
            for response in result['responses']:
                self.assertIn('style', response)
                self.assertIn('content', response)
                self.assertIn('confidence', response)
    
    def test_session_management(self):
        """Test session management integration"""
        # Process multiple queries in same session
        session_id = None
        
        queries = [
            "What is property law?",
            "How does it apply to marriage?",
            "What about divorce proceedings?"
        ]
        
        for query in queries:
            result = self.rag_system.process_query(query, session_id)
            
            if session_id is None:
                session_id = result['session_id']
            
            self.assertEqual(result['session_id'], session_id)
        
        # Check session info
        session_info = self.rag_system.get_session_info(session_id)
        self.assertIn('conversation_history', session_info)
        self.assertEqual(len(session_info['conversation_history']), 3)
    
    def test_document_search(self):
        """Test direct document search"""
        query = "commercial contract law"
        results = self.rag_system.search_legal_documents(query, top_k=5)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 5)

class TestSystemIntegration(unittest.TestCase):
    """Test System Integration and Performance"""
    
    def setUp(self):
        self.rag_system = RAGSystem()
    
    def test_performance_benchmarks(self):
        """Test system performance benchmarks"""
        test_queries = [
            "property ownership rights in Sri Lanka",
            "divorce procedures and requirements",
            "employment contract termination",
            "commercial dispute resolution",
            "child custody laws"
        ]
        
        total_time = 0
        successful_queries = 0
        
        for query in test_queries:
            start_time = time.time()
            
            result = self.rag_system.process_query(query)
            
            end_time = time.time()
            query_time = end_time - start_time
            total_time += query_time
            
            if result['success']:
                successful_queries += 1
                self.assertLess(query_time, 10.0)  # Should complete within 10 seconds
        
        avg_time = total_time / len(test_queries)
        success_rate = successful_queries / len(test_queries)
        
        print(f"\nPerformance Results:")
        print(f"Average query time: {avg_time:.2f}s")
        print(f"Success rate: {success_rate:.2%}")
        
        self.assertGreater(success_rate, 0.8)  # At least 80% success rate
        self.assertLess(avg_time, 5.0)  # Average under 5 seconds
    
    def test_error_handling(self):
        """Test error handling and recovery"""
        # Test with invalid session
        result = self.rag_system.get_session_info("invalid_session_id")
        self.assertIn('error', result)
        
        # Test with empty query
        result = self.rag_system.process_query("")
        self.assertIsInstance(result, dict)
    
    def test_concurrent_sessions(self):
        """Test handling multiple concurrent sessions"""
        sessions = []
        
        # Create multiple sessions
        for i in range(5):
            result = self.rag_system.process_query(f"Test query {i}")
            sessions.append(result['session_id'])
        
        # Verify all sessions are unique
        self.assertEqual(len(set(sessions)), 5)
        
        # Verify all sessions are active
        for session_id in sessions:
            session_info = self.rag_system.get_session_info(session_id)
            self.assertNotIn('error', session_info)

def run_comprehensive_tests():
    """Run all tests and generate report"""
    print("=" * 60)
    print("🧪 RAG SYSTEM COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestVectorDBConnector,
        TestResponseGenerator,
        TestDialogManager,
        TestRAGSystem,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    successes = total_tests - failures - errors
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {successes}")
    print(f"❌ Failed: {failures}")
    print(f"🔥 Errors: {errors}")
    print(f"Success Rate: {(successes/total_tests)*100:.1f}%")
    
    if failures > 0:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if errors > 0:
        print("\n🔥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    print("\n" + "=" * 60)
    
    if failures == 0 and errors == 0:
        print("🎉 ALL TESTS PASSED! RAG System is ready for production.")
    else:
        print("⚠️  Some tests failed. Please review and fix issues before deployment.")
    
    return result

if __name__ == "__main__":
    run_comprehensive_tests()
