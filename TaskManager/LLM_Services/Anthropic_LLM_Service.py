from Base_LLM_Service import BaseLLMService
from anthropic import Anthropic
import CommonUtils

class OpenAI_LLm_Service(BaseLLMService):
    def __init__(self, api_key, model):
        self.client = Anthropic(api_key)
        self.model = model

    def form_message(self, content, role):
        message = {"role": role, "content": content}

        return message
    
    def get_function_schemas(self):
        function_schemas = []
        for function in CommonUtils.function_map:
            fs = CommonUtils.GetFunctionSchemaForAnthropic(CommonUtils.function_map[function])
            function_schemas.append(fs)

        return function_schemas
    

    def send_message(self, user_input, messages):
        response = self.client.responses.create(
            model=self.model,
            input=messages,
            tools=self.get_function_schemas(),
            tool_choice="auto"
        )
        return response
