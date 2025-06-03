import json
import re
from typing import List, Dict, Any, Optional, Literal

from app.infrastructure.llm.base_llm import BaseLLM # Updated import

class QueryClassifier:
    def __init__(self, llm_client: BaseLLM): # Updated type hint
        self.llm_client = llm_client

    def _parse_llm_classification_response(self, response_text: str) -> Dict[str, Any]:
        try:
            response_text = response_text.strip()
            # Attempt to match single word responses first, as per simplified prompt
            response_lower_single_word = response_text.lower()
            if response_lower_single_word == "greeting":
                return {"type": "greeting", "confidence": 0.9, "explanation": "LLM classified as greeting."}
            if response_lower_single_word == "irrelevant":
                return {"type": "irrelevant", "confidence": 0.9, "explanation": "LLM classified as irrelevant."}
            if response_lower_single_word == "relevant":
                return {"type": "relevant", "confidence": 0.9, "explanation": "LLM classified as relevant."}

            # Fallback to JSON parsing if single word match fails (e.g., if LLM doesn't follow instructions strictly)
            if response_text.startswith("{") and response_text.endswith("}"):
                data = json.loads(response_text)
                return {
                    "type": data.get("type", "relevant").lower(),
                    "confidence": data.get("confidence", 0.7),
                    "explanation": data.get("explanation", "")
                }

            # Fallback to keyword search in a more verbose response
            response_lower_verbose = response_text.lower()
            if "greeting" in response_lower_verbose: return {"type": "greeting", "confidence": 0.85, "explanation": "LLM classified as greeting (verbose fallback)."}
            if "irrelevant" in response_lower_verbose: return {"type": "irrelevant", "confidence": 0.85, "explanation": "LLM classified as irrelevant (verbose fallback)."}
            if "relevant" in response_lower_verbose: return {"type": "relevant", "confidence": 0.85, "explanation": "LLM classified as relevant (verbose fallback)."}

            return {"type": "relevant", "confidence": 0.6, "explanation": "LLM response unclear, defaulting to relevant."}
        except json.JSONDecodeError:
            print(f"QueryClassifier: Could not parse LLM JSON response: {response_text}")
            response_lower = response_text.lower() # Re-check original non-JSON text
            if "greeting" in response_lower: return {"type": "greeting", "confidence": 0.8, "explanation": "LLM classified as greeting (non-JSON fallback)."}
            if "irrelevant" in response_lower: return {"type": "irrelevant", "confidence": 0.8, "explanation": "LLM classified as irrelevant (non-JSON fallback)."}
            return {"type": "relevant", "confidence": 0.5, "explanation": "LLM response not clearly parsable (JSON error), defaulting to relevant."}
        except Exception as e:
            print(f"QueryClassifier: Error parsing LLM response: {e}")
            return {"type": "relevant", "confidence": 0.5, "explanation": f"Error parsing LLM response: {e}"}

    def classify_query(self, query_text: str, context_docs_content: Optional[List[str]] = None) -> Dict[str, Any]:
        prompt = self._create_classification_prompt(query_text, context_docs_content)
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.llm_client.chat(messages=messages) # Removed model_name, assume client default or pre-config
            response_text = ""
            if isinstance(response, str): response_text = response
            elif hasattr(response, 'content'): response_text = response.content # Common for some client objects
            elif isinstance(response, dict) and 'choices' in response and response['choices']: # OpenAI like
                message_content = response['choices'][0].get('message', {}).get('content')
                if message_content: response_text = message_content

            if not response_text: # Check if response_text was successfully extracted
                 print("QueryClassifier: LLM response was empty or in unexpected format after attempting to extract content.")
                 return self._fallback_classification(query_text, context_docs_content)
            return self._parse_llm_classification_response(response_text)
        except Exception as e:
            print(f"Error in LLM classification call: {e}")
            return self._fallback_classification(query_text, context_docs_content)

    def _create_classification_prompt(self, query_text: str, context_docs_content: Optional[List[str]]):
        prompt_lines = [
            "Your task is to classify the user's query.",
            f'User Query: "{query_text}"'
        ]
        if context_docs_content:
            # Limit context size to avoid overly long prompts
            context_preview = "\n\n".join(context_docs_content)[:2000] # Max 2000 chars for context preview
            prompt_lines.append(f"Relevant Context from Documents (preview):\n---BEGIN CONTEXT---\n{context_preview}\n---END CONTEXT---\n")

        prompt_lines.extend([
            'Based on the query and any provided context, classify the query into ONE of the following types: "greeting", "relevant", "irrelevant".',
            '- "greeting": If the query is simple small talk, a greeting, a salutation, or a simple thank you.',
            '- "relevant": If the query is asking for information that could potentially be found in related documents OR if no context is provided but the query is substantive and seeking information.',
            '- "irrelevant": If context is provided AND the query is clearly unrelated to the provided context OR if the query is nonsensical/off-topic.',
            'Respond with ONLY ONE WORD selected from ["greeting", "relevant", "irrelevant"]. Do not add any other text or punctuation.',
            'For example, if the query is asking about the capital of France and context is about LLMs, respond: irrelevant',
            'If the query is "hello", respond: greeting',
            'If the query is "what is a large language model" and context is about LLMs, respond: relevant'
        ])
        return "\n".join(prompt_lines)

    def _fallback_classification(self, query_text: str, context_docs_content: Optional[List[str]] = None): # Added default None
        if self._basic_is_greeting(query_text):
            return {"type": "greeting", "confidence": 0.8, "explanation": "Fallback: Query appears to be a greeting"}

        # Only consider irrelevant if context is actually provided
        if context_docs_content is not None: # Explicitly check if context was given
            if self._basic_is_irrelevant(query_text, context_docs_content):
                return {"type": "irrelevant", "confidence": 0.6, "explanation": "Fallback: Query appears unrelated to provided context"}

        # Default to relevant if not a greeting and (no context given OR not clearly irrelevant to given context)
        return {"type": "relevant", "confidence": 0.7, "explanation": "Fallback: Query appears to be a relevant question"}

    def _basic_is_greeting(self, query_text: str) -> bool:
        greeting_phrases = ['hi','hello','hey','greetings','good morning','good afternoon','good evening','how are you',"what's up",'howdy','hola','thanks','thank you', 'bye', 'goodbye']
        query_lower = query_text.lower().strip()
        # Remove punctuation for more robust matching
        query_lower_no_punct = re.sub(r'[^\w\s]', '', query_lower)

        for phrase in greeting_phrases:
            # Check for exact match or if query starts/ends with the phrase (with or without punctuation)
            if query_lower_no_punct == phrase or \
               query_lower_no_punct.startswith(phrase + ' ') or \
               query_lower_no_punct.endswith(' ' + phrase) or \
               query_lower_no_punct == phrase: # Handles single word greetings like "hello"
                return True

        words = query_lower_no_punct.split()
        # Consider short queries (<=3 words) containing a greeting word as greetings
        return len(words) <= 3 and any(word in greeting_phrases for word in words)

    def _basic_is_irrelevant(self, query_text: str, context_docs_content: List[str]) -> bool:
        if not context_docs_content: # If there's no context, it's hard to judge irrelevance this way
            return False # Default to not irrelevant if no context is provided to compare against

        stop_words = {'a','an','the','and','or','but','is','are','was','were','in','on','at','to','for','with','by','about','like','from','of','as','this','that','these','those','it','its', 'do', 'does', 'what', 'how', 'who', 'when', 'where', 'why'}
        query_words = {word for word in query_text.lower().split() if word not in stop_words and len(word) > 2}

        if not query_words: return False # If query is all stop words or very short words, less likely to be irrelevant by this metric

        combined_context = " ".join(context_docs_content).lower()
        matches = sum(1 for word in query_words if word in combined_context)

        # Consider irrelevant if less than 20% of significant query words are in context
        return (matches / len(query_words)) < 0.20

    def get_classification(self, query_text: str, context_docs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        context_content_list = [doc.get("page_content", "") for doc in context_docs] if context_docs else None
        return self.classify_query(query_text, context_content_list)

    def get_greeting_response(self) -> str:
        # Consider a list of varied greeting responses
        greetings = [
            "Hello! How can I help you with your documents today?",
            "Hi there! What can I do for you regarding your documents?",
            "Greetings! I'm here to assist with your document-related questions."
        ]
        import random
        return random.choice(greetings)

    def get_irrelevant_question_response(self) -> str:
        return "I am sorry, but your question seems unrelated to the documents I have access to. My primary function is to answer questions based on the content of the provided documents."
