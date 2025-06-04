from typing import List, Dict, Any, Callable
from UI.UI_Components import UIComponents


class ChatInterface:
    """Chat interface component for the RAG application"""
    
    def __init__(self, on_message_callback: Callable[[str], Dict[str, Any]]):
        """
        Initialize the chat interface
        
        Args:
            on_message_callback: Callback function to handle new messages
        """
        self.on_message_callback = on_message_callback
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
        UIComponents.create_subheader_UI("Chat with your Documents")
        
        # Display chat history
        self._render_chat_history()
        
        # Chat input
        self._render_chat_input()
    
    def _render_chat_history(self):
        """Render the chat history"""
        for message in UIComponents.get_session_state_messages:
            UIComponents.display_message_with_role(message["role"], message["content"])
    
    def _render_chat_input(self):
        """Render the chat input"""
        if prompt := UIComponents.chat_input("Ask a question about your documents"):
            # Add user message to UI
            UIComponents.get_session_state_messages.append({"role": "user", "content": prompt})
            UIComponents.display_message_with_role("user", prompt)

            
            # Process message if documents are loaded
            if UIComponents.get_session_state_variable("documents", None):
                with UIComponents.display_chat_message_with_role("assistant"):
                    with UIComponents.display_spinner("🤔 Thinking..."):
                        # Get response from callback
                        self.display_message(None)
            else:
                UIComponents.display_error("Please upload and process documents first.")
    
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
