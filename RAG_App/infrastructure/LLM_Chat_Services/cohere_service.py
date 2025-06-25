from .base_llm_service import BaseLLMService
import cohere
import infrastructure.common.RAG_Constants as constants
from infrastructure.common.component_registry import register, LLM_SERVICES_REGISTRY

@register(LLM_SERVICES_REGISTRY, name=constants.LLMServiceType.COHERE.value)
class CohereService(BaseLLMService):
    def __init__(self, api_key: str, model_name: str):
        self.client = cohere.Client(api_key=api_key)
        self.model_name = model_name

    def generate_response(self, prompt, **kwargs):
        response = self.client.chat(
            model=self.model_name,
            message=prompt 
        )
        return response.text