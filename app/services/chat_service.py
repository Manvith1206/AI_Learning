from app.infrastructure.llm.base_llm import BaseLLM
from app.services.document_service import DocumentService # Corrected: DocumentService is the one being passed
from app.config import settings
from typing import List, Dict, Any, Optional

from .query_classifier import QueryClassifier # Assuming QueryClassifier is in the same directory

class ChatService:
    def __init__(self, llm_client: BaseLLM, document_service: DocumentService): # document_service is the one passed
        self.llm_client = llm_client
        self.document_service = document_service # This is DocumentService instance
        self.query_classifier = QueryClassifier(llm_client)

    def get_chat_response(self, user_query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        # Initial quick check for greeting without full classification to save resources
        if self.query_classifier._basic_is_greeting(user_query):
            # If basic check suggests greeting, get full classification to be sure
            # Pass no context for pure greeting classification
            classification_result = self.query_classifier.get_classification(user_query, context_docs=None)
            if classification_result["type"] == "greeting":
                return {"answer": self.query_classifier.get_greeting_response(), "type": "greeting", "retrieved_docs": []}

        retrieved_docs: List[Dict[str, Any]] = [] # Ensure type consistency
        try:
            # Use the DocumentService to retrieve documents. DocumentService handles its own retriever/reranker.
            retrieved_docs = self.document_service.retrieve_relevant_documents(query=user_query)
        except Exception as e:
            print(f"Error retrieving documents in ChatService: {e}")
            # Proceed without RAG context if retrieval fails, but log the error.
            pass

        # Classify the query using retrieved documents (if any) as context
        classification_result = self.query_classifier.get_classification(user_query, context_docs=retrieved_docs)

        if classification_result["type"] == "irrelevant":
            return {"answer": self.query_classifier.get_irrelevant_question_response(), "type": "irrelevant", "retrieved_docs": retrieved_docs}

        # If relevant (or fallback to relevant), proceed with RAG
        context_for_llm = "\n\n".join([doc.get("page_content", "") for doc in retrieved_docs]) if retrieved_docs else "No relevant context was found in the documents."

        messages: List[Dict[str, str]] = []

        system_prompt = getattr(settings, 'DEFAULT_SYSTEM_PROMPT',
                                "You are a helpful AI assistant. Your primary function is to answer questions based on the provided context from documents. "
                                "If the context does not provide the answer, clearly state that the information is not found in the documents. "
                                "Do not use external knowledge or make assumptions beyond the provided text.")
        messages.append({"role": "system", "content": system_prompt})

        if chat_history: # Add chat history if provided
            messages.extend(chat_history)

        # Construct the RAG prompt for the user's turn
        rag_prompt_parts = [
            "Please consider the following context derived from the available documents:",
            "---BEGIN DOCUMENT CONTEXT---",
            context_for_llm,
            "---END DOCUMENT CONTEXT---",
            "",
            f"User Query: {user_query}",
            "",
            "Based solely on the document context provided, please answer the user's query. "
            "If the answer cannot be found in the context, explicitly state that."
        ]
        messages.append({"role": "user", "content": "\n".join(rag_prompt_parts)})

        try:
            response_data = self.llm_client.chat(messages=messages) # model_name is assumed to be part of llm_client's config

            llm_answer = ""
            # Standardize response extraction
            if isinstance(response_data, str):
                llm_answer = response_data
            elif hasattr(response_data, 'content') and isinstance(response_data.content, str): # For objects with a 'content' attribute
                llm_answer = response_data.content
            elif isinstance(response_data, dict) and 'choices' in response_data and response_data['choices']: # OpenAI-like
                choice = response_data['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    llm_answer = choice['message']['content']
                elif 'text' in choice: # Older OpenAI completion format or other models
                    llm_answer = choice['text']

            if not llm_answer: # If no content could be extracted
                 llm_answer = "Sorry, I received an empty response from the language model."

            return {"answer": llm_answer, "type": "chat", "retrieved_docs": retrieved_docs}
        except Exception as e:
            print(f"Error during final LLM call in ChatService: {e}")
            return {"answer": "Sorry, I encountered an error while generating a response.", "type": "error", "retrieved_docs": retrieved_docs}
