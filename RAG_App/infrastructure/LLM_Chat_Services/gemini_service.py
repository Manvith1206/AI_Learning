import time
from .base_llm_service import BaseLLMService
from infrastructure.Common.RAG_Constants import GeminiLLMModel
from google.genai import types
from google import genai

class GeminiService(BaseLLMService):
    def __init__(self, client, model_name=GeminiLLMModel.GEMINI_FLASH.value):
        self.client = client
        self.model_name = model_name
        self.cost = 0
        self.time_taken = 0
    def generate_response(self, prompt, **kwargs):
        start_time = time.time()
        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1
            )
        ):
            print("Yielded Chunks", chunk)
            if hasattr(chunk, 'text'):
                yield chunk.text

        end_time = time.time()
        self.time_taken = end_time - start_time
    
    def function_call(self, functions, prompt, **kwargs):
        tools = [types.Tool(function_declarations=[func]) for func in functions]

        config = types.GenerateContentConfig(
            tools=tools,
            temperature=kwargs.get("temperature", 0.7)
        )

        response = self.client.models.generate_content(
        model=self.model_name,
        contents=prompt,
        config=config
    )
        return response
    
    def get_function_args(self, response):
        if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                # For newer LLM APIs that return structured function calls
                                function_args = part.function_call.args
                                return function_args
                            
    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken
    
    def get_function_schema(self):
        return {
            "name": "classify_query",
            "description": "Classify a user query as greeting, relevant, or irrelevant",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["greeting", "relevant", "irrelevant"],
                        "description": "The type of query"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score between 0 and 1"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation for the classification"
                    }
                },
                "required": ["query_type", "confidence", "explanation"]
            }
        }