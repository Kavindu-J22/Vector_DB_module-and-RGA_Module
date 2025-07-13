"""
Awesome Frontend for RAG Legal Assistant
Beautiful Streamlit-based chat interface
"""

import streamlit as st
import time
import json
from datetime import datetime
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
from rag_system import get_rag_system
import config as rag_config

# Page configuration
st.set_page_config(
    page_title=rag_config.FRONTEND_TITLE,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .response-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .response-card.recommended {
        border-left: 5px solid #28a745;
        background: #f8fff9;
    }
    
    .response-card.professional {
        border-left: 5px solid #007bff;
    }
    
    .response-card.detailed {
        border-left: 5px solid #ffc107;
    }
    
    .response-card.concise {
        border-left: 5px solid #17a2b8;
    }
    
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    
    .user-message {
        background: #e3f2fd;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: #f3e5f5;
        margin-right: 2rem;
    }
    
    .source-item {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.25rem 0;
        font-size: 0.9em;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = get_rag_system()
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = st.session_state.rag_system.dialog_manager.create_session()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_responses' not in st.session_state:
        st.session_state.current_responses = []
    
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'preferred_style': 'professional',
            'language': 'english',
            'detail_level': 'medium'
        }

def render_header():
    """Render the main header"""
    st.markdown(f"""
    <div class="main-header">
        <h1>⚖️ {rag_config.FRONTEND_TITLE}</h1>
        <p>{rag_config.FRONTEND_SUBTITLE}</p>
        <p><em>Your AI-powered legal research assistant for Sri Lankan law</em></p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with controls and information"""
    with st.sidebar:
        st.header("🔧 Settings")
        
        # User preferences
        st.subheader("Response Preferences")
        
        preferred_style = st.selectbox(
            "Preferred Response Style",
            ["professional", "detailed", "concise"],
            index=["professional", "detailed", "concise"].index(st.session_state.user_preferences['preferred_style'])
        )
        
        detail_level = st.selectbox(
            "Detail Level",
            ["high", "medium", "low"],
            index=["high", "medium", "low"].index(st.session_state.user_preferences['detail_level'])
        )
        
        # Update preferences if changed
        if (preferred_style != st.session_state.user_preferences['preferred_style'] or 
            detail_level != st.session_state.user_preferences['detail_level']):
            
            st.session_state.user_preferences.update({
                'preferred_style': preferred_style,
                'detail_level': detail_level
            })
            
            st.session_state.rag_system.dialog_manager.update_user_preferences(
                st.session_state.session_id, 
                st.session_state.user_preferences
            )
        
        st.divider()
        
        # System status
        st.subheader("📊 System Status")
        
        try:
            status = st.session_state.rag_system.get_system_status()
            
            if status['overall_status'] == 'operational':
                st.success("🟢 System Operational")
            else:
                st.error("🔴 System Issues")
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", f"{status['vector_database']['total_vectors']:,}")
            with col2:
                st.metric("Dimension", status['vector_database']['dimension'])
            
        except Exception as e:
            st.error(f"Status Error: {e}")
        
        st.divider()
        
        # Session info
        st.subheader("💬 Session Info")
        st.info(f"Session ID: {st.session_state.session_id[:8]}...")
        st.info(f"Messages: {len(st.session_state.chat_history)}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.session_state.current_responses = []
            st.session_state.session_id = st.session_state.rag_system.dialog_manager.create_session()
            st.rerun()

def render_response_card(response: Dict[str, Any], index: int, is_history: bool = False):
    """Render a single response card"""
    
    # Determine card class based on style and recommendation
    card_class = f"response-card {response['style']}"
    if response.get('recommended', False):
        card_class += " recommended"
    
    # Confidence styling
    confidence = response['confidence']
    confidence_class = f"confidence-{confidence}"
    
    # Recommendation badge
    recommendation_badge = "🌟 RECOMMENDED" if response.get('recommended', False) else ""
    
    st.markdown(f"""
    <div class="{card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h4>{response['title']} {recommendation_badge}</h4>
            <span class="{confidence_class}">Confidence: {confidence.upper()}</span>
        </div>
        <div style="margin-bottom: 1rem;">
            {response['content']}
        </div>
        <div style="font-size: 0.9em; color: #666;">
            <strong>Sources:</strong> {response['num_sources']} documents | 
            <strong>Style:</strong> {response['style'].capitalize()} |
            <strong>Rank:</strong> #{response.get('rank', index + 1)}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sources expander
    if response['sources']:
        with st.expander(f"📚 View {len(response['sources'])} Sources"):
            for i, source in enumerate(response['sources']):
                st.markdown(f"""
                <div class="source-item">
                    <strong>Source {i+1}:</strong> {source['title']}<br>
                    <em>Type:</em> {source['type']} | 
                    <em>Category:</em> {source['category']} | 
                    <em>Relevance:</em> {source['score']:.3f}
                </div>
                """, unsafe_allow_html=True)
    
    # Feedback and action buttons
    if not is_history:
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button(f"👍 Helpful", key=f"helpful_{index}"):
                st.session_state.rag_system.update_response_feedback(
                    st.session_state.session_id,
                    len(st.session_state.chat_history),
                    index,
                    "helpful"
                )
                st.success("Thank you for your feedback!")

        with col2:
            if st.button(f"👎 Not Helpful", key=f"not_helpful_{index}"):
                st.session_state.rag_system.update_response_feedback(
                    st.session_state.session_id,
                    len(st.session_state.chat_history),
                    index,
                    "not_helpful"
                )
                st.info("Feedback recorded. We'll improve!")

        with col3:
            if st.button(f"📋 Copy Response", key=f"copy_{index}"):
                st.code(response['content'], language=None)
    else:
        st.info("💡 This is a response from your conversation history.")
        if st.button(f"📋 Copy Response", key=f"copy_history_{index}"):
            st.code(response['content'], language=None)

def render_chat_interface():
    """Render the main chat interface"""
    
    # Chat input
    st.subheader("💬 Ask Your Legal Question")
    
    # Query input
    # Check if we need to set a new example
    if 'selected_example' in st.session_state:
        # Use the selected example and clear it
        default_value = st.session_state.selected_example
        del st.session_state.selected_example
    else:
        default_value = ""

    user_query = st.text_area(
        "Enter your legal question:",
        value=default_value,
        placeholder=rag_config.CHAT_PLACEHOLDER,
        height=100,
        key="user_input"
    )
    
    # Advanced options
    with st.expander("🔧 Advanced Search Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            doc_type_filter = st.selectbox(
                "Document Type",
                ["Any", "act", "case"],
                index=0
            )
        
        with col2:
            category_filter = st.selectbox(
                "Legal Category", 
                ["Any", "Family", "Property", "Commercial", "Labour"],
                index=0
            )
    
    # Submit button
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        submit_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)
    
    with col2:
        example_button = st.button("💡 Example", use_container_width=True)
    
    with col3:
        search_button = st.button("📚 Search Only", use_container_width=True)
    
    # Handle example button
    if example_button:
        st.session_state.show_examples = True
        st.rerun()

    # Show example selection if requested
    if st.session_state.get('show_examples', False):
        st.markdown("### 💡 Example Legal Questions")
        examples = [
            "What are the property ownership rights in Sri Lanka?",
            "How to file for divorce in Sri Lankan courts?",
            "What are the employment termination procedures?",
            "Commercial contract dispute resolution process",
            "Child custody laws in Sri Lanka"
        ]

        col_ex1, col_ex2 = st.columns([3, 1])
        with col_ex1:
            selected_example = st.selectbox("Choose an example:", examples, key="example_selector")
        with col_ex2:
            if st.button("Use This Example", key="use_example"):
                st.session_state.selected_example = selected_example
                st.session_state.show_examples = False
                st.rerun()
            if st.button("Cancel", key="cancel_example"):
                st.session_state.show_examples = False
                st.rerun()
    
    # Handle search only button
    if search_button and user_query:
        with st.spinner("🔍 Searching legal documents..."):
            filters = {}
            if doc_type_filter != "Any":
                filters['document_type'] = doc_type_filter
            if category_filter != "Any":
                filters['primary_category'] = category_filter
            
            search_results = st.session_state.rag_system.search_legal_documents(
                user_query, filters, top_k=10
            )
            
            st.subheader(f"📚 Search Results ({len(search_results)} documents found)")
            
            for i, result in enumerate(search_results):
                with st.expander(f"Document {i+1}: {result['filename']} (Score: {result['score']:.3f})"):
                    st.write(f"**Type:** {result['document_type']}")
                    st.write(f"**Category:** {result['category']}")
                    st.write(f"**Content Preview:**")
                    st.write(result['content'])
    
    # Handle main submit
    if submit_button and user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_query,
            'timestamp': time.time()
        })
        
        # Process query
        with st.spinner("🤔 Analyzing your question and generating responses..."):
            
            # Prepare filters
            filters = {}
            if doc_type_filter != "Any":
                filters['document_type'] = doc_type_filter
            if category_filter != "Any":
                filters['primary_category'] = category_filter
            
            # Process query with RAG system
            result = st.session_state.rag_system.process_query(
                user_query,
                st.session_state.session_id,
                st.session_state.user_preferences,
                filters
            )
            
            if result['success']:
                st.session_state.current_responses = result['responses']
                
                # Add assistant message to chat history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': result['responses'],
                    'timestamp': time.time(),
                    'processing_time': result['processing_time'],
                    'docs_count': result['retrieved_docs_count']
                })
                
                st.success(f"✅ Generated {len(result['responses'])} responses in {result['processing_time']:.2f}s")
            else:
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        # Input will be cleared on next rerun
        st.rerun()

def render_responses():
    """Render the current responses"""
    
    if st.session_state.current_responses:
        st.subheader("🎯 AI-Generated Responses")
        st.info("💡 **Tip:** We've generated 3 different response styles. Choose the one that best suits your needs!")
        
        # Response selection tabs
        tab1, tab2, tab3 = st.tabs([
            f"🏛️ {st.session_state.current_responses[0]['title']}" + (" ⭐" if st.session_state.current_responses[0].get('recommended') else ""),
            f"📖 {st.session_state.current_responses[1]['title']}" + (" ⭐" if st.session_state.current_responses[1].get('recommended') else ""),
            f"⚡ {st.session_state.current_responses[2]['title']}" + (" ⭐" if st.session_state.current_responses[2].get('recommended') else "")
        ])
        
        with tab1:
            render_response_card(st.session_state.current_responses[0], 0)
        
        with tab2:
            render_response_card(st.session_state.current_responses[1], 1)
        
        with tab3:
            render_response_card(st.session_state.current_responses[2], 2)
        
        # Legal disclaimer
        st.warning(rag_config.LEGAL_DISCLAIMER)

    # Display selected history response
    if hasattr(st.session_state, 'selected_history_response') and st.session_state.selected_history_response:
        st.subheader("📜 Selected Response from History")
        render_response_card(st.session_state.selected_history_response, -1, is_history=True)

        # Clear button
        if st.button("Clear Selected Response", key="clear_history_response"):
            del st.session_state.selected_history_response
            st.rerun()

def render_chat_history():
    """Render chat history with improved UI"""

    if st.session_state.chat_history:
        st.subheader("💬 Conversation History")

        # Limit display to recent messages
        recent_history = st.session_state.chat_history[-rag_config.MAX_CHAT_HISTORY_DISPLAY:]

        for i, message in enumerate(recent_history):
            timestamp = datetime.fromtimestamp(message['timestamp']).strftime("%H:%M:%S")

            if message['role'] == 'user':
                # User message with attractive styling
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 15px;
                    border-radius: 15px 15px 5px 15px;
                    margin: 10px 0;
                    margin-left: 20%;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 18px; margin-right: 8px;">👤</span>
                        <strong>You</strong>
                        <span style="margin-left: auto; font-size: 12px; opacity: 0.8;">{timestamp}</span>
                    </div>
                    <div style="font-size: 14px; line-height: 1.4;">
                        {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:  # assistant
                processing_time = message.get('processing_time', 0)
                docs_count = message.get('docs_count', 0)
                responses = message.get('content', [])

                # Assistant message with clickable responses
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                    padding: 15px;
                    border-radius: 15px 15px 15px 5px;
                    margin: 10px 0;
                    margin-right: 20%;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 18px; margin-right: 8px;">🤖</span>
                        <strong>Legal Assistant</strong>
                        <span style="margin-left: auto; font-size: 12px; opacity: 0.8;">{timestamp}</span>
                    </div>
                    <div style="font-size: 12px; opacity: 0.9; margin-bottom: 10px;">
                        Generated {len(responses)} responses in {processing_time:.2f}s using {docs_count} legal documents
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Show clickable response previews
                if responses:
                    st.markdown("**Click on a response to view it below:**")
                    cols = st.columns(min(len(responses), 3))

                    for idx, response in enumerate(responses):
                        with cols[idx % 3]:
                            response_preview = response.get('content', '')[:100] + "..." if len(response.get('content', '')) > 100 else response.get('content', '')

                            if st.button(
                                f"📋 {response.get('title', f'Response {idx+1}')}",
                                key=f"history_response_{i}_{idx}",
                                help=response_preview,
                                use_container_width=True
                            ):
                                # Set the selected response to display
                                st.session_state.selected_history_response = response
                                st.rerun()

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        render_chat_interface()
        
        # Current responses
        render_responses()
    
    with col2:
        # Chat history
        render_chat_history()
        
        # Quick stats
        if st.session_state.current_responses:
            st.subheader("📊 Response Analytics")
            
            # Confidence distribution
            confidences = [r['confidence'] for r in st.session_state.current_responses]
            confidence_counts = {c: confidences.count(c) for c in set(confidences)}
            
            fig = px.pie(
                values=list(confidence_counts.values()),
                names=list(confidence_counts.keys()),
                title="Response Confidence Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
