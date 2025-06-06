from typing import List, Dict, Any, Callable
from UI.UI_Components import UIComponents
import Utils.Utils

class ChatInterface:
    """Chat interface component for the RAG application"""
    
    def __init__(self, pipeline, on_message_callback: Callable[[str], Dict[str, Any]]):
        """
        Initialize the chat interface
        
        Args:
            on_message_callback: Callback function to handle new messages
        """
        self.on_message_callback = on_message_callback
        self.pipeline = pipeline
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables for chat"""
        UIComponents.initialize_session_state({
            "messages": [],
            "documents": None,
            "chunks": None
        })
    
    def render(self):
        """Render the chat interface"""
        
        # Display chat history
        self.render_chat_history()
        
        # Chat input
        self.render_chat_input()
    
    def render_chat_history(self):
        """Render the chat history"""
        for message in UIComponents.get_session_state_messages():
            UIComponents.display_message_with_role(message["role"], message["content"])
    
    def render_chat_input(self):
        """Render the chat input"""
        UIComponents.create_subheader_UI("Chat with your Documents")
        
        # Display chat history
        # for message in UIComponents.get_session_state_messages():
        #     UIComponents.display_message_with_role(role=message["role"], message=message['content'])

        print("Chat Messages:", UIComponents.get_session_state_messages())
        # Chat input
        import streamlit as st
        if prompt := UIComponents.chat_input("Ask a question about your documents", key="chat_input"):
            UIComponents.add_message_to_chat(role='user', content=prompt)
            UIComponents.display_message_with_role(role='user', message=prompt)
            UIComponents.process_chat_input(role='assistant', content=prompt, pipeline=self.pipeline, prompt=prompt)
    
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
