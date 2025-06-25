from typing import List, Dict, Any, Callable
from UI.UI_Components import UIComponents


from infrastructure.common.rag_pipeline import RAGPipeline

class ChatInterface:
    """Chat interface component for the RAG application"""
    
    def __init__(self, pipeline: RAGPipeline):
        """
        Initialize the chat interface
        
        Args:
            pipeline: The RAG pipeline instance.
        """
        self.pipeline = pipeline
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables for chat"""
        UIComponents.initialize_session_state({
            "messages": [],
            "documents": None,
            "chunks": None
        })
    
    def render(self):
        """Render the chat interface"""
        
        # Display chat history
        self._render_chat_history()
        
        # Chat input
        self._render_chat_input()
    
    def _render_chat_history(self):
        """Render the chat history"""
        for message in UIComponents.get_session_state_messages():
            UIComponents.display_message_with_role(message["role"], message["content"])
    
    def _render_chat_input(self):
        """Render the chat input"""
        UIComponents.create_subheader_UI("Chat with your Documents")
        
        # Display chat history
        # for message in UIComponents.get_session_state_messages():
        #     UIComponents.display_message_with_role(role=message["role"], message=message['content'])

        # Chat input
        import streamlit as st
        if prompt := UIComponents.chat_input("Ask a question about your documents", key="chat_input"):
            UIComponents.add_message_to_chat(role='user', content=prompt)
            UIComponents.display_message_with_role(role='user', message=prompt)
            
            with UIComponents.display_chat_message_with_role(role='assistant', message=""):
                with UIComponents.display_spinner("Thinking..."):
                    history_text = "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in UIComponents.get_session_state_messages()])
                    response_stream = self.pipeline.query_processing.query(prompt, history_text=history_text)
                    
                    full_response = ""
                    is_rerank_explanation_rendered = False
                    empty_placeholder = UIComponents.create_empty_placeholder()
                    
                    for delta in response_stream:
                        if delta.get("rerank_explanation") and not is_rerank_explanation_rendered:
                            UIComponents.create_subheader_UI(f"**Re-ranking Explanation:**\n{delta['rerank_explanation']}")
                            is_rerank_explanation_rendered = True
                        
                        full_response = delta.get("answer", "")
                        empty_placeholder.markdown(full_response)
                        
                    UIComponents.add_message_to_chat('assistant', full_response)
    
    def display_message(self, response):
        """Display the response message in the chat interface"""        
        if response is not None:
            # Display rerank explanation if available
            if "rerank_explanation" in response:
                UIComponents.create_subheader_UI(f"**Re-ranking Explanation:**\n{response['rerank_explanation']}")
            
            # Display answer
            UIComponents.create_subheader_UI(response["answer"])
            
            # Add to message history
            UIComponents.get_session_state_messages.append({
                "role": "assistant",
                "content": response["answer"]
            })

    def clear_history(self):
        """Clear the chat history"""
        UIComponents.get_session_state_messages = []
