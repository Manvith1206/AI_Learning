from .base_llm_service import BaseLLMService
from rag_modular.Common.RAG_Constants import GeminiLLMModel
from google.genai import types

class GeminiService(BaseLLMService):
    def __init__(self, client, model_name=GeminiLLMModel.GEMINI_FLASH.value):
        self.client = client
        self.model_name = model_name
    def generate_response(self, prompt, **kwargs):
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text
    
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
        