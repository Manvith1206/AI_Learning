from .base_prompt_provider import BasePromptProvider
import infrastructure.prompt_templates.flashcards_generation_prompt_templates as prompt_templates

class FlashCardsGeneration_Prompt_Provider(BasePromptProvider):
    def get_system_prompt(self):
        return prompt_templates.SYSTEM_PROMPT
    def get_user_prompt(self, kwargs):
        return self._format_prompt(prompt_templates.GENERATE_FLASHCARDS_USER_PROMPT, **kwargs)
    def get_final_prompt(self, kwargs):
        system_prompt = self.get_system_prompt()
        user_prompt = self.get_user_prompt(**kwargs)
        return system_prompt + "\n" + user_prompt