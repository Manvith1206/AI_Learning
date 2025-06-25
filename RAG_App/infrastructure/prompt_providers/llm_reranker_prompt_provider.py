from .base_prompt_provider import BasePromptProvider
import infrastructure.prompt_templates.llm_reranker_prompt_templates as prompt_templates

class LLM_Reranker_Prompt_Provider(BasePromptProvider):
    def get_user_prompt(self, **kwargs):
        return self._format_prompt(prompt_templates.LLM_RERANK_PROMPT, **kwargs)
    
    def get_final_prompt(self, **kwargs):
        user_prompt = self.get_user_prompt(**kwargs)
        return user_prompt