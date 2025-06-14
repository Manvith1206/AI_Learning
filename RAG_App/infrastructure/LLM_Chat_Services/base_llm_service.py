from abc import ABC, abstractmethod

class BaseLLMService(ABC):
    @abstractmethod
    def generate_response(self, prompt, **kwargs):
        pass
    @abstractmethod
    def get_cost_and_time_taken(self):
        """
        Get the cost and time taken for the LLM service call
        """
        pass
    @abstractmethod
    def get_function_schema(self):
        pass
    @abstractmethod
    def get_function_args(self):
        pass
    @abstractmethod
    def function_call(self, functions, prompt, **kwargs):
        pass
