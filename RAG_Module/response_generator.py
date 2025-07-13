"""
Response Generator for RAG Module
Generates multiple response variants using different strategies
"""

# import openai  # Commented out due to dependency issues
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import config as rag_config

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generates legal responses using RAG approach with multiple variants"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or rag_config.OPENAI_API_KEY
        if self.api_key and self.api_key != "your-openai-api-key-here":
            # openai.api_key = self.api_key  # Commented out due to dependency issues
            self.llm_available = False  # Set to False for now
            logger.warning("OpenAI integration disabled. Using template-based response generation.")
        else:
            self.llm_available = False
            logger.warning("OpenAI API key not configured. Using fallback response generation.")
    
    def generate_responses(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                          conversation_history: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Generate multiple response variants for the given query"""
        
        if not retrieved_docs:
            return self._generate_no_results_responses(query)
        
        # Determine confidence level based on retrieval scores
        confidence_level = self._assess_confidence(retrieved_docs)
        
        # Generate three different response variants
        responses = []
        
        # Response 1: Professional and Comprehensive
        professional_response = self._generate_professional_response(query, retrieved_docs, confidence_level)
        responses.append({
            'style': 'professional',
            'title': 'Professional Legal Analysis',
            'content': professional_response,
            'confidence': confidence_level,
            'sources': self._extract_sources(retrieved_docs[:3])
        })
        
        # Response 2: Detailed and Explanatory
        detailed_response = self._generate_detailed_response(query, retrieved_docs, confidence_level)
        responses.append({
            'style': 'detailed',
            'title': 'Detailed Legal Explanation',
            'content': detailed_response,
            'confidence': confidence_level,
            'sources': self._extract_sources(retrieved_docs[:5])
        })
        
        # Response 3: Concise and Direct
        concise_response = self._generate_concise_response(query, retrieved_docs, confidence_level)
        responses.append({
            'style': 'concise',
            'title': 'Concise Legal Summary',
            'content': concise_response,
            'confidence': confidence_level,
            'sources': self._extract_sources(retrieved_docs[:2])
        })
        
        # Add metadata to all responses
        for response in responses:
            response.update({
                'timestamp': time.time(),
                'query': query,
                'num_sources': len(response['sources']),
                'disclaimer': rag_config.LEGAL_DISCLAIMER
            })
        
        return responses
    
    def _generate_professional_response(self, query: str, docs: List[Dict[str, Any]], 
                                      confidence: str) -> str:
        """Generate a professional legal response"""
        
        if self.llm_available:
            return self._generate_llm_response(query, docs, "professional", confidence)
        else:
            return self._generate_template_response(query, docs, "professional", confidence)
    
    def _generate_detailed_response(self, query: str, docs: List[Dict[str, Any]], 
                                  confidence: str) -> str:
        """Generate a detailed explanatory response"""
        
        if self.llm_available:
            return self._generate_llm_response(query, docs, "detailed", confidence)
        else:
            return self._generate_template_response(query, docs, "detailed", confidence)
    
    def _generate_concise_response(self, query: str, docs: List[Dict[str, Any]], 
                                 confidence: str) -> str:
        """Generate a concise direct response"""
        
        if self.llm_available:
            return self._generate_llm_response(query, docs, "concise", confidence)
        else:
            return self._generate_template_response(query, docs, "concise", confidence)
    
    def _generate_llm_response(self, query: str, docs: List[Dict[str, Any]], 
                              style: str, confidence: str) -> str:
        """Generate response using OpenAI LLM"""
        
        try:
            # Prepare context from retrieved documents
            context = self._prepare_context(docs, style)
            
            # Create style-specific prompts
            prompts = {
                "professional": f"""
                As a legal AI assistant specializing in Sri Lankan law, provide a professional legal analysis for the following question.
                
                Question: {query}
                
                Legal Context:
                {context}
                
                Instructions:
                - Provide a professional, well-structured legal analysis
                - Reference specific legal provisions and cases when available
                - Use formal legal language appropriate for legal professionals
                - Include relevant legal principles and precedents
                - Confidence level: {confidence}
                - If confidence is low, clearly state limitations
                
                Response:""",
                
                "detailed": f"""
                As a legal AI assistant, provide a comprehensive explanation of the following legal question for someone seeking to understand Sri Lankan law.
                
                Question: {query}
                
                Legal Context:
                {context}
                
                Instructions:
                - Provide a detailed, educational explanation
                - Break down complex legal concepts into understandable parts
                - Explain the reasoning behind legal principles
                - Include examples where helpful
                - Use clear, accessible language while maintaining accuracy
                - Confidence level: {confidence}
                
                Response:""",
                
                "concise": f"""
                As a legal AI assistant, provide a clear and direct answer to the following legal question about Sri Lankan law.
                
                Question: {query}
                
                Legal Context:
                {context}
                
                Instructions:
                - Provide a concise, direct answer
                - Focus on the most important legal points
                - Use clear, straightforward language
                - Avoid unnecessary legal jargon
                - Confidence level: {confidence}
                - If uncertain, state this clearly
                
                Response:"""
            }
            
            # Generate response using OpenAI (disabled for now)
            # response = openai.ChatCompletion.create(
            #     model=rag_config.DEFAULT_MODEL,
            #     messages=[
            #         {"role": "system", "content": "You are a legal AI assistant specializing in Sri Lankan law."},
            #         {"role": "user", "content": prompts[style]}
            #     ],
            #     max_tokens=rag_config.MAX_TOKENS,
            #     temperature=rag_config.TEMPERATURE
            # )
            #
            # return response.choices[0].message.content.strip()

            # Fallback to template response for now
            return self._generate_template_response(query, docs, style, confidence)
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return self._generate_template_response(query, docs, style, confidence)
    
    def _generate_template_response(self, query: str, docs: List[Dict[str, Any]], 
                                  style: str, confidence: str) -> str:
        """Generate response using templates (fallback when LLM not available)"""
        
        # Extract key information from documents
        relevant_content = []
        document_types = set()
        categories = set()
        
        for doc in docs[:3]:
            relevant_content.append(doc.get('content', ''))
            document_types.add(doc.get('document_type', 'unknown'))
            categories.add(doc.get('category', 'unknown'))
        
        # Combine content
        combined_content = ' '.join(relevant_content)[:1000]  # Limit length
        
        # Generate style-specific responses
        if style == "professional":
            response = f"""
**Legal Analysis**

Based on the retrieved legal documents from Sri Lankan law, the following analysis addresses your query: "{query}"

**Relevant Legal Framework:**
{combined_content}

**Legal Assessment:**
The available legal sources indicate that this matter falls under {', '.join(categories)} law, with relevant provisions found in {', '.join(document_types)}. 

**Conclusion:**
{self._get_confidence_statement(confidence)} The legal framework provides guidance on this matter, though specific circumstances may require individual legal consultation.

**Document Types Referenced:** {', '.join(document_types)}
**Legal Categories:** {', '.join(categories)}
"""
        
        elif style == "detailed":
            response = f"""
**Comprehensive Legal Explanation**

Your question "{query}" relates to Sri Lankan legal provisions that I'll explain in detail below.

**Background and Context:**
This legal matter involves {', '.join(categories)} law, which is governed by various {', '.join(document_types)} in Sri Lankan jurisprudence.

**Detailed Analysis:**
{combined_content}

**Key Legal Principles:**
1. The legal framework establishes clear guidelines for such matters
2. Relevant precedents and statutory provisions provide guidance
3. Specific circumstances may affect the application of these principles

**Practical Implications:**
{self._get_confidence_statement(confidence)} Understanding these legal principles is important for proper compliance and decision-making.

**Sources:** Based on {len(docs)} relevant legal documents
"""
        
        else:  # concise
            response = f"""
**Direct Answer**

Regarding "{query}":

{combined_content[:300]}...

**Key Point:** {self._get_confidence_statement(confidence)} The legal framework addresses this matter under {', '.join(categories)} law.

**Sources:** {len(docs)} relevant legal documents reviewed.
"""
        
        return response.strip()
    
    def _prepare_context(self, docs: List[Dict[str, Any]], style: str) -> str:
        """Prepare context from retrieved documents"""
        
        context_parts = []
        max_docs = 5 if style == "detailed" else 3
        
        for i, doc in enumerate(docs[:max_docs]):
            content = doc.get('content', '')
            doc_type = doc.get('document_type', 'unknown')
            filename = doc.get('filename', 'unknown')
            
            context_part = f"""
Document {i+1} ({doc_type}): {filename}
Content: {content}
---"""
            context_parts.append(context_part)
        
        return '\n'.join(context_parts)
    
    def _assess_confidence(self, docs: List[Dict[str, Any]]) -> str:
        """Assess confidence level based on retrieval scores"""
        
        if not docs:
            return "low"
        
        # Calculate average score
        scores = [doc.get('score', 0) for doc in docs[:3]]
        avg_score = sum(scores) / len(scores)
        
        if avg_score >= rag_config.CONFIDENCE_THRESHOLDS["high"]:
            return "high"
        elif avg_score >= rag_config.CONFIDENCE_THRESHOLDS["medium"]:
            return "medium"
        else:
            return "low"
    
    def _get_confidence_statement(self, confidence: str) -> str:
        """Get appropriate confidence statement"""
        
        statements = {
            "high": "Based on strong legal precedents and clear statutory provisions,",
            "medium": "According to the available legal sources,",
            "low": "Based on limited available information,"
        }
        
        return statements.get(confidence, "Based on the available legal documents,")
    
    def _extract_sources(self, docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract source information from documents"""
        
        sources = []
        for doc in docs:
            source = {
                'id': doc.get('id', ''),
                'title': doc.get('filename', 'Unknown Document'),
                'type': doc.get('document_type', 'unknown'),
                'category': doc.get('category', 'unknown'),
                'score': round(doc.get('score', 0), 3)
            }
            sources.append(source)
        
        return sources
    
    def _generate_no_results_responses(self, query: str) -> List[Dict[str, Any]]:
        """Generate responses when no relevant documents are found"""
        
        base_response = f"""
I apologize, but I couldn't find specific legal documents in the Sri Lankan legal database that directly address your query: "{query}"

This could be due to:
1. The query may be too specific or use terminology not present in the database
2. The legal matter might fall outside the scope of the current document collection
3. The query might need to be rephrased for better matching

**Suggestions:**
- Try rephrasing your question using different legal terms
- Break down complex questions into simpler parts
- Consider broader legal categories that might be relevant

**Important:** For specific legal advice, please consult with a qualified legal professional familiar with Sri Lankan law.
"""
        
        responses = []
        for style in ["professional", "detailed", "concise"]:
            responses.append({
                'style': style,
                'title': f'{style.capitalize()} Response - No Results Found',
                'content': base_response,
                'confidence': 'low',
                'sources': [],
                'timestamp': time.time(),
                'query': query,
                'num_sources': 0,
                'disclaimer': rag_config.LEGAL_DISCLAIMER
            })
        
        return responses
