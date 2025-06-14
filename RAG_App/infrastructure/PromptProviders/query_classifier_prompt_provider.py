from .Base_Prompt_Provider import BasePromptProvider
import infrastructure.PromptTemplates.query_classifier_prompt_templates as prompt_templates

class Query_Classifier_Prompt_Provider(BasePromptProvider):
    def get_base_prompt(self, **kwargs):
        return self._format_prompt(prompt_templates.BASE_QUERY_CLASSIFER_PROMPT, **kwargs)
    def get_prompt_with_contexts(self, **kwargs):
        return self._format_prompt(prompt_templates.QUERY_CLASSIFIER_WITH_CONTEXTS_PROMPT, **kwargs)
    def get_prompt_without_contexts(self):
        return prompt_templates.QUERY_CLASSIFIER_WITHOUT_CONTEXTS_PROMPT
    