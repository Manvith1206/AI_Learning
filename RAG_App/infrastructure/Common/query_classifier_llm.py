"""
Query classifier module using LLM function calling to detect greetings and irrelevant questions in a RAG system.
"""
import json
from typing import List, Dict, Any, Optional, Literal
from infrastructure.prompt_providers.query_classifier_prompt_provider import Query_Classifier_Prompt_Provider
from infrastructure.llm_chat_services.base_llm_service import BaseLLMService

class QueryClassifier:
    """
    Classifies user queries into different types using LLM function calling:
    - Greetings
    - Irrelevant questions
    - Relevant questions (default)
    """
    
    def __init__(self, llm_service: BaseLLMService=None):
        """
        Initialize the query classifier.
        
        Args:
            llm_service: LLM service to use for classification (will be set by RAGPipeline)
        """
        self.llm_service = llm_service
        self.prompt_provider = Query_Classifier_Prompt_Provider()
    
    def set_llm_service(self, llm_service):
        """
        Set the LLM service to use for classification.
        
        Args:
            llm_service: LLM service instance
        """
        self.llm_service = llm_service
    
    def classify_query(self, query_text: str, context_docs: Optional[List[str]] = None):
        """
        Classify a query using LLM function calling.
        
        Args:
            query_text (str): The user's query text
            context_docs (List[str], optional): Retrieved context documents
            
        Returns:
            Dict[str, Any]: Classification result with type and confidence
        """
        if not self.llm_service:
            # Fallback to basic classification if LLM service is not available
            if self._basic_is_greeting(query_text):
                return {"type": "greeting", "confidence": 0.9}
            elif context_docs and self._basic_is_irrelevant(query_text, context_docs):
                return {"type": "irrelevant", "confidence": 0.7}
            else:
                return {"type": "relevant", "confidence": 0.8}
        
        # Define the function schema for the LLM
        function_schema = self.llm_service.get_function_schema()
        
        # Create the prompt for classification
        prompt = self._create_classification_prompt(query_text, context_docs)
        
        # Call the LLM with function calling
        try:
            print("LLM Service: ", self.llm_service)
            response = self.llm_service.function_call(
                functions=[function_schema],
                prompt=prompt
            )
            print("FunctionCall Response: ", response)

            # Parse the response
            function_args = self.llm_service.get_function_args(response)
            print("FunctionCall / FunctionArgs: ", function_args)
            
            if isinstance(function_args, str):
                function_args = json.loads(function_args)
            return {
                "type": function_args.get("query_type", "relevant"),
                "confidence": function_args.get("confidence", 0.5),
                "explanation": function_args.get("explanation", "")
            }
            
        except Exception as e:
            print(f"Error in LLM classification: {str(e)}, {str(e.__traceback__)}")
            return self._fallback_classification(query_text, context_docs)
    
    def _create_classification_prompt(self, query_text: str, context_docs: Optional[List[str]]):
        """
        Create a prompt for the LLM to classify the query.
        
        Args:
            query_text (str): The user's query
            context_docs (List[str], optional): Retrieved context documents
            
        Returns:
            str: Prompt for the LLM
        """
        prompt = self.prompt_provider.get_base_prompt(query_text=query_text)
        
        if context_docs:
            context_text = "\n\n".join(context_docs)
            prompt += self.prompt_provider.get_prompt_with_contexts(context_text=context_text)
        else:
            prompt += self.prompt_provider.get_prompt_without_contexts()
        
        return prompt
    
    def _fallback_classification(self, query_text: str, context_docs: Optional[List[str]]):
        """
        Fallback classification method when LLM fails.
        
        Args:
            query_text (str): The user's query
            context_docs (List[str], optional): Retrieved context documents
            
        Returns:
            Dict[str, Any]: Classification result
        """
        if self._basic_is_greeting(query_text):
            return {"type": "greeting", "confidence": 0.8, "explanation": "Query appears to be a greeting"}
        elif context_docs and self._basic_is_irrelevant(query_text, context_docs):
            return {"type": "irrelevant", "confidence": 0.6, "explanation": "Query appears unrelated to context"}
        else:
            return {"type": "relevant", "confidence": 0.7, "explanation": "Query appears to be a relevant question"}
    
    def _basic_is_greeting(self, query_text: str):
        """
        Basic method to check if a query is a greeting.
        
        Args:
            query_text (str): The user's query
            
        Returns:
            bool: True if the query is likely a greeting
        """
        greeting_phrases = [
            'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
            'good evening', 'how are you', 'what\'s up', 'howdy', 'hola'
        ]
        
        query_lower = query_text.lower()
        # Check for exact matches or if query starts with greeting
        for phrase in greeting_phrases:
            if query_lower == phrase or query_lower.startswith(phrase + ' '):
                return True
        
        # Check if query is very short and might be a greeting
        words = query_lower.split()
        return len(words) <= 3 and any(word in greeting_phrases for word in words)
    
    def _basic_is_irrelevant(self, query_text: str, context_docs: List[str]):
        """
        Basic method to check if a query is irrelevant to the context.
        
        Args:
            query_text (str): The user's query
            context_docs (List[str]): Retrieved context documents
            
        Returns:
            bool: True if the query appears irrelevant to the context
        """
        # If no context was retrieved, the query might be irrelevant
        if not context_docs or len(context_docs) == 0:
            return True
        
        # Extract significant words from query (excluding common stop words)
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
            'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 
            'from', 'of', 'as', 'this', 'that', 'these', 'those', 'it', 'its'
        }
        
        # Get significant words from query
        query_words = set()
        for word in query_text.lower().split():
            if word not in stop_words and len(word) > 2:
                query_words.add(word)
        
        # Skip check if query is too short or only has common words
        if len(query_words) < 2:
            return False
        
        # Check if significant query words appear in the context
        combined_context = " ".join(context_docs).lower()
        matches = sum(1 for word in query_words if word in combined_context)
        
        # If less than 30% of query terms are found in context, consider it irrelevant
        return matches / len(query_words) < 0.3 if query_words else False
    
    def is_greeting(self, query_text: str):
        """
        Check if the query is a greeting using LLM classification.
        
        Args:
            query_text (str): The user's query text
            
        Returns:
            bool: True if the query is a greeting, False otherwise
        """
        result = self.classify_query(query_text)
        return result["type"] == "greeting"
    
    def is_irrelevant(self, query_text: str, context_docs: List[str]):
        """
        Check if the query is irrelevant to the context using LLM classification.
        
        Args:
            query_text (str): The user's query text
            context_docs (List[str]): Retrieved context documents
            
        Returns:
            bool: True if the query is irrelevant, False otherwise
        """
        result = self.classify_query(query_text, context_docs)
        return result["type"] == "irrelevant"
    
    def get_greeting_response(self) -> str:
        """
        Generate a friendly greeting response.
        
        Returns:
            str: A greeting response
        """
        return """Hello! I'm your document assistant. I can help answer questions about the documents you've uploaded. 
        
How can I help you today? If you have any specific questions about your documents, feel free to ask!"""
    
    def get_irrelevant_question_response(self):
        """
        Generate a response for irrelevant questions.
        
        Returns:
            str: A response for irrelevant questions
        """
        return """I'm sorry, but your question appears to be unrelated to the documents I have access to. 

I'm designed to help answer questions specifically about the content in your uploaded documents. Could you please ask a question related to the documents, or upload additional documents if you're looking for information on a different topic?"""
