"""
Task Manager application with modular AI service architecture.
This module provides the main entry point for the Task Manager application.
"""
import streamlit as st
import json
from datetime import datetime
import sys
import os

# Import local modules
import CommonUtils
from UIManager import DisplayCurrentTasks, DisplayTaskManager
from Functions import add_task, add_task_with_subtasks_and_tags, complete_task, add_subtasks, tag_tasks, delete_task, get_tasks_based_on_tag

# Import AI service modules
from ai_services.service_manager import ServiceManager
from ai_services.service_factory import get_available_services

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "tasks" not in st.session_state:
    st.session_state.tasks = []

if "service_type" not in st.session_state:
    st.session_state.service_type = "openai"

# Initialize service manager
service_manager = ServiceManager(service_type=st.session_state.service_type)

# Register functions for function calling
service_manager.register_functions([
    add_task,
    add_task_with_subtasks_and_tags,
    complete_task,
    add_subtasks,
    tag_tasks,
    delete_task,
    get_tasks_based_on_tag
])

def call_llm(user_input):
    """
    Call the AI service with the user input.
    
    Args:
        user_input: The user input
    """
    # Add system message if not present
    if not any(msg.get("role") == "system" for msg in st.session_state.messages):
        st.session_state.messages.append({
            "role": "system", 
            "content": "You are a task planner that decomposes user requests into multiple function calls."
        })
    
    # Call the AI service
    result = service_manager.call_llm(
        user_input=user_input,
        messages=st.session_state.messages,
        model=None  # Use default model for the service
    )
    
    # Update messages
    st.session_state.messages = result["messages"]
    
    # Display current tasks and messages
    DisplayCurrentTasks(st.session_state.tasks)
    for message in st.session_state.messages:
        # Handle standard message types (user/assistant)
        if isinstance(message, dict) and "role" in message:
            if message["role"] != "system":    
                if isinstance(message['content'], str):
                    with st.chat_message(message["role"]):
                        st.markdown(message['content'])
        # Log any other message formats
        else:
            print(f"Unhandled message format: {type(message)}")

def main():
    """Main entry point for the application."""
    st.title("Task Manager :u7533:")
    
    # Service selection in sidebar
    st.sidebar.title("AI Service Settings")
    available_services = get_available_services()
    selected_service = st.sidebar.selectbox(
        "Select AI Service",
        available_services,
        index=available_services.index(st.session_state.service_type)
    )
    
    # Change service if selection changed
    if selected_service != st.session_state.service_type:
        st.session_state.service_type = selected_service
        service_manager.change_service(selected_service)
        st.session_state.messages = []  # Clear messages when changing service
        st.rerun()
    print("Service:", selected_service)

    # # API key input
    # api_key = st.sidebar.text_input(
    #     f"{selected_service.capitalize()} API Key",
    #     type="password",
    #     value=service_manager.config_manager.get_api_key(selected_service) or ""
    # )
    
    # # Save API key if provided
    # if api_key and api_key != service_manager.config_manager.get_api_key(selected_service):
    #     service_manager.config_manager.set_api_key(selected_service, api_key)
    #     service_manager.service.initialize(api_key)
    
    # Display task manager interface
    DisplayTaskManager()

if __name__ == "__main__":
    main()
