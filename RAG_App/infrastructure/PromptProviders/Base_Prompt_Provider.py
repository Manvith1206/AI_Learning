class BasePromptProvider:
    def _format_prompt(self, prompt: str, **kwargs):
        return prompt.format(**kwargs)
    def get_system_prompt(self):
        pass