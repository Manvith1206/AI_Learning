from .Base_Prompt_Provider import BasePromptProvider
import infrastructure.PromptTemplates.LLM_Evaluation_prompt_templates as prompt_templates

class LLM_Evaluation_Prompt_Provider(BasePromptProvider):
    def get_faithfullness_prompt(self, **kwargs):
        return self._format_prompt(prompt_templates.FAITHFULLNESS_CALCULATION_PROMPT_TEMPLATE, **kwargs)
    
    def get_context_recall_prompt(self, **kwargs):
        return self._format_prompt(prompt_templates.CONTEXT_RECALL_CALCUALTION_PROMPT_TEMPLATE, **kwargs)
    
    def get_context_precision_prompt(self, **kwargs):
        return self._format_prompt(prompt_templates.CONTEXT_PRECISION_CALCUALTION_PROMPT_TEMPLATE, **kwargs)
    
    def get_answer_relavancy_prompt(self, **kwargs):
        return self._format_prompt(prompt_templates.ANSWER_RELAVANCY_CALCULATION_PROMPT_TEMPLATE, **kwargs)