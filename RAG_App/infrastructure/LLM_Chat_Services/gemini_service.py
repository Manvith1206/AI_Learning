import time
from .base_llm_service import BaseLLMService
from infrastructure.common.rag_constants import GeminiLLMModel, LLMServiceType
from google.generativeai import types
from google import genai
from infrastructure.common.component_registry import register, LLM_SERVICES_REGISTRY

@register(LLM_SERVICES_REGISTRY, name=LLMServiceType.GEMINI.value)
class GeminiService(BaseLLMService):
    def __init__(self, api_key: str, model_name=GeminiLLMModel.GEMINI_FLASH.value):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.cost = 0
        self.time_taken = 0
        
    def generate_response(self, prompt, **kwargs):
        start_time = time.time()
        generation_config = types.GenerationConfig(
            temperature=kwargs.get("temperature", 0.1)
        )
        stream = self.model.generate_content(
            contents=prompt,
            generation_config=generation_config,
            stream=True
        )
        last_chunk = None
        for chunk in stream:
            if hasattr(chunk, 'text'):
                yield chunk.text
            last_chunk = chunk

        if last_chunk and hasattr(last_chunk, 'usage_metadata'):
            total_tokens = last_chunk.usage_metadata.total_token_count
            self.cost = self.get_cost_based_on_model(total_tokens)

        end_time = time.time()
        self.time_taken = end_time - start_time
        
    def get_cost_based_on_model(self, tokens):
        if self.model_name == GeminiLLMModel.GEMINI_FLASH.value:
            return (tokens / 1000000) * 0.80
        elif self.model_name == GeminiLLMModel.GEMINI_PRO.value:
            if tokens <= 200000:
                return (tokens / 1000000) * 10
            else:
                return (tokens / 1000000) * 15
        elif self.model_name == GeminiLLMModel.GEMINI_TWO_5_FLASH.value:
            return 0
        return 0
        
    def function_call(self, functions, prompt, **kwargs):
        tools = [types.Tool(function_declarations=[func for func in functions])]
        generation_config = types.GenerationConfig(
            temperature=kwargs.get("temperature", 0.7)
        )

        response = self.model.generate_content(
            contents=prompt,
            tools=tools,
            generation_config=generation_config
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