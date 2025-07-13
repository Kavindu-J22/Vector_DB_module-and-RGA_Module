"""
Vector Database Connector for RAG Module
Connects to the existing Vector Database and Embedding Module
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pinecone import Pinecone
import hashlib
import logging
from typing import List, Dict, Any, Optional
import config as rag_config

logger = logging.getLogger(__name__)

class VectorDBConnector:
    """Connector to the existing Vector Database for document retrieval"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=rag_config.PINECONE_API_KEY)
        self.index = self.pc.Index(rag_config.PINECONE_INDEX_NAME)
        logger.info("VectorDBConnector initialized")
    
    def create_query_embedding(self, query: str, dimension: int = rag_config.VECTOR_DIMENSION) -> List[float]:
        """Create embedding for query using same method as document processing"""
        # Create a hash-based embedding (same as used in document processing)
        text_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Convert hash to numbers
        hash_numbers = [int(text_hash[i:i+2], 16) for i in range(0, len(text_hash), 2)]
        
        # Normalize to [-1, 1] range
        normalized = [(x - 127.5) / 127.5 for x in hash_numbers]
        
        # Extend or truncate to desired dimension
        if len(normalized) < dimension:
            multiplier = dimension // len(normalized) + 1
            extended = (normalized * multiplier)[:dimension]
        else:
            extended = normalized[:dimension]
        
        # Add text-based features
        words = query.lower().split()
        word_count = len(words)
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        
        # Modify embedding based on text features
        for i in range(min(10, len(extended))):
            extended[i] += (word_count % 100) / 1000.0
            extended[i] += (avg_word_length % 10) / 100.0
        
        # Ensure values are in reasonable range
        extended = [max(-1.0, min(1.0, x)) for x in extended]
        
        return extended
    
    def search_documents(self, query: str, top_k: int = rag_config.RETRIEVAL_TOP_K, 
                        filters: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Search for relevant documents using vector similarity"""
        try:
            # Create query embedding
            query_embedding = self.create_query_embedding(query)
            
            # Build filter dictionary
            filter_dict = {}
            if filters:
                for key, value in filters.items():
                    if value:
                        filter_dict[key] = value
            
            # Perform search
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            # Format results
            results = []
            for match in search_results.get('matches', []):
                result = {
                    'id': match['id'],
                    'score': match['score'],
                    'content': match['metadata'].get('text_preview', ''),
                    'metadata': match.get('metadata', {}),
                    'document_type': match['metadata'].get('document_type', 'unknown'),
                    'category': match['metadata'].get('primary_category', 'unknown'),
                    'filename': match['metadata'].get('filename', 'unknown'),
                    'sequence_number': match['metadata'].get('sequence_number', 0)
                }
                results.append(result)
            
            logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def hybrid_search(self, query: str, top_k: int = rag_config.RETRIEVAL_TOP_K,
                     alpha: float = 0.7) -> List[Dict[str, Any]]:
        """Hybrid search combining vector similarity and keyword matching"""
        try:
            # Get vector search results
            vector_results = self.search_documents(query, top_k * 2)
            
            # Simple keyword matching for hybrid scoring
            query_words = set(query.lower().split())
            
            # Re-score results with hybrid approach
            for result in vector_results:
                vector_score = result['score']
                
                # Calculate keyword overlap score
                content_words = set(result['content'].lower().split())
                keyword_overlap = len(query_words.intersection(content_words)) / len(query_words.union(content_words))
                
                # Combine scores
                hybrid_score = alpha * vector_score + (1 - alpha) * keyword_overlap
                result['hybrid_score'] = hybrid_score
                result['keyword_score'] = keyword_overlap
            
            # Sort by hybrid score and return top_k
            hybrid_results = sorted(vector_results, key=lambda x: x['hybrid_score'], reverse=True)
            return hybrid_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return self.search_documents(query, top_k)
    
    def get_document_context(self, doc_id: str, context_window: int = 3) -> List[Dict[str, Any]]:
        """Get surrounding document chunks for better context"""
        try:
            # Extract base document ID and sequence number
            parts = doc_id.split('_')
            if len(parts) >= 3:
                base_id = '_'.join(parts[:-1])
                current_seq = int(parts[-1])
                
                # Search for related chunks
                context_results = []
                for seq_offset in range(-context_window, context_window + 1):
                    target_seq = current_seq + seq_offset
                    if target_seq >= 0:
                        target_id = f"{base_id}_{target_seq:04d}"
                        
                        # Try to fetch this chunk
                        try:
                            result = self.index.fetch(ids=[target_id])
                            if target_id in result.get('vectors', {}):
                                vector_data = result['vectors'][target_id]
                                context_results.append({
                                    'id': target_id,
                                    'content': vector_data['metadata'].get('text_preview', ''),
                                    'metadata': vector_data['metadata'],
                                    'sequence_number': target_seq
                                })
                        except:
                            continue
                
                return sorted(context_results, key=lambda x: x['sequence_number'])
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting document context: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', 0),
                'index_fullness': stats.get('index_fullness', 0)
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def filter_by_confidence(self, results: List[Dict[str, Any]], 
                           threshold: float = rag_config.MIN_SIMILARITY_THRESHOLD) -> List[Dict[str, Any]]:
        """Filter results by confidence threshold"""
        filtered_results = []
        for result in results:
            score = result.get('hybrid_score', result.get('score', 0))
            if score >= threshold:
                filtered_results.append(result)
        
        logger.info(f"Filtered {len(results)} results to {len(filtered_results)} above threshold {threshold}")
        return filtered_results
    
    def rerank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Re-rank results based on additional criteria"""
        try:
            # Simple re-ranking based on document type preference and recency
            def rerank_score(result):
                base_score = result.get('hybrid_score', result.get('score', 0))
                
                # Boost recent documents (higher sequence numbers)
                seq_boost = min(result.get('sequence_number', 0) / 100.0, 0.1)
                
                # Boost certain document types for legal queries
                type_boost = 0.0
                if result.get('document_type') == 'act':
                    type_boost = 0.05  # Slight boost for acts
                elif result.get('document_type') == 'case':
                    type_boost = 0.03  # Slight boost for cases
                
                # Category relevance boost
                category_boost = 0.0
                query_lower = query.lower()
                category = result.get('category', '').lower()
                
                if any(word in query_lower for word in ['property', 'ownership', 'land']):
                    if category == 'property':
                        category_boost = 0.1
                elif any(word in query_lower for word in ['marriage', 'divorce', 'family']):
                    if category == 'family':
                        category_boost = 0.1
                elif any(word in query_lower for word in ['contract', 'commercial', 'business']):
                    if category == 'commercial':
                        category_boost = 0.1
                elif any(word in query_lower for word in ['employment', 'labor', 'work']):
                    if category == 'labour':
                        category_boost = 0.1
                
                return base_score + seq_boost + type_boost + category_boost
            
            # Re-rank and return
            reranked = sorted(results, key=rerank_score, reverse=True)
            logger.info(f"Re-ranked {len(results)} results")
            return reranked
            
        except Exception as e:
            logger.error(f"Error re-ranking results: {e}")
            return results
