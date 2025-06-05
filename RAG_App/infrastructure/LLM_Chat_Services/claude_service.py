import json
import time
from .base_llm_service import BaseLLMService
import anthropic
import infrastructure.Common.RAG_Constants as constants

class ClaudeService(BaseLLMService):
    def __init__(self, client, model_name="claude-2"):
        self.client = client
        self.model_name = model_name
        self.time_taken = 0
        self.cost = 0
        
    def generate_response(self, prompt, **kwargs):
        start_time = time.time()
        print(f"Generating response with Claude model: {self.model_name}")
        with self.client.messages.stream(
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
        ) as stream:
            for text in stream.text_stream:
                yield text
        
        end_time = time.time()
        self.time_taken = end_time - start_time
        
    def get_cost_and_time_taken(self):
        """
        Get the cost and time taken for the Claude service call
        """
        return self.cost, self.time_taken
    
    def function_call(self, functions, prompt, **kwargs):
        response = self.client.messages.create(
            model=self.model_name,
            messages=[
            {
                "role": "user",
                "content": prompt  # your actual prompt content
            }
        ],
            tools=functions,
            max_tokens=kwargs.get('max_tokens', 1024)
        )
        
        return response
    
    def get_function_args(self, response):
        if hasattr(response, 'content'):
            for content_item in response.content:
                if hasattr(content_item, 'type') and content_item.type == "tool_use":
                    # Create a tool call object similar to our common format
                    function_args = json.dumps(content_item.input)
                    return function_args
                
    def get_cost_based_on_model(self, tokens):
        if self.model == constants.CLAUDE_MODELS.CLAUDE_HAIKU_THREE_5.value:
            return (tokens / 1000000) * 0.80
        elif self.model == constants.CLAUDE_MODELS.CLAUDE_SONNET_THREE_7.value:
            return (tokens / 1000000) * 3
        elif self.model == constants.CLAUDE_MODELS.CLAUDE_SONNET_THREE_5.value:
            return (tokens / 1000000) * 3
        elif self.model == constants.CLAUDE_MODELS.CLAUDE_OPUS_THREE.value:
            return (tokens / 1000000) * 15
        
    def get_function_schema(self):
        return {
            "name": "classify_query",
            "description": "Classify a user query as greeting, relevant, or irrelevant",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["greeting", "relevant", "irrelevant"],
                        "description": "The type of query"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score between 0 and 1"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation for the classification"
                    }
                },
                "required": ["query_type", "confidence", "explanation"]
            }
        }