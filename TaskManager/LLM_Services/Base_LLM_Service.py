from abc import ABC, abstractmethod

class BaseLLMService():
    def __init__(self, api_key, model):
        pass
    @abstractmethod
    def form_message(self, content, role):
        pass
    @abstractmethod
    def send_message(self, user_input, tools, messages):
        pass
    
    @abstractmethod
    def get_function_schemas(self):
        pass