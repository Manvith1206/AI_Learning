from typing import List, Dict, Any, Optional, Union, Iterator
from app.infrastructure.llm.base_llm import BaseLLM
# from app.config import settings # For API key source during instantiation

# Ensure openai and tiktoken are in requirements.txt
try:
    from openai import OpenAI, APIError
    # from openai.types.chat import ChatCompletionChunk # For streaming type hint if desired
    # from openai.types.chat.chat_completion import ChatCompletion # For non-streaming return type
except ImportError:
    print("openai library not found. Please install it.")
    OpenAI = None # type: ignore
    APIError = Exception # type: ignore
    # ChatCompletionChunk = Any
    # ChatCompletion = Any

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    print("tiktoken library not found. Please install it for token counting.")
    TIKTOKEN_AVAILABLE = False
    tiktoken = None # type: ignore

class OpenAIClient(BaseLLM):
    def __init__(self,
                 api_key: str,
                 chat_model_name: str = "gpt-3.5-turbo",
                 embedding_model_name: str = "text-embedding-ada-002",
                 **kwargs: Any):
        super().__init__() # Call parent __init__
        if OpenAI is None:
            raise ImportError("openai library is required to use OpenAIClient.")
        if not api_key:
            raise ValueError("OpenAI API key must be provided.")

        self.client = OpenAI(api_key=api_key)
        self.chat_model_name = chat_model_name
        self.embedding_model_name = embedding_model_name
        self.default_temperature = kwargs.get("temperature", 0.7)
        self.default_max_tokens = kwargs.get("max_tokens") # Default to None unless specified

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs: Any
    ) -> Union[str, Iterator[str], Any]:

        # Prepare arguments for OpenAI API call
        api_params: Dict[str, Any] = {
            "model": kwargs.get("model_name", self.chat_model_name),
            "messages": messages, # type: ignore
            "stream": stream,
            "temperature": kwargs.get("temperature", self.default_temperature),
        }

        # Add optional parameters if they are provided in kwargs or have defaults
        if kwargs.get("max_tokens", self.default_max_tokens) is not None:
            api_params["max_tokens"] = kwargs.get("max_tokens", self.default_max_tokens)
        if kwargs.get("top_p") is not None:
            api_params["top_p"] = kwargs.get("top_p")
        if kwargs.get("presence_penalty") is not None:
            api_params["presence_penalty"] = kwargs.get("presence_penalty")
        if kwargs.get("frequency_penalty") is not None:
            api_params["frequency_penalty"] = kwargs.get("frequency_penalty")
        if kwargs.get("tools") is not None:
            api_params["tools"] = kwargs.get("tools")
        if kwargs.get("tool_choice") is not None:
            api_params["tool_choice"] = kwargs.get("tool_choice")

        try:
            completion = self.client.chat.completions.create(**api_params) # type: ignore
        except APIError as e:
            print(f"OpenAI API Error in chat: {e}")
            raise RuntimeError(f"OpenAI API Error: {str(e)}") from e
        except Exception as e:
            print(f"An unexpected error occurred in OpenAIClient.chat: {e}")
            raise RuntimeError(f"Unexpected error in OpenAI chat: {str(e)}") from e

        if stream:
            def stream_iterator():
                for chunk in completion: # completion is an iterator of ChatCompletionChunk
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
            return stream_iterator()
        else:
            if hasattr(completion, 'choices') and completion.choices and completion.choices[0].message:
                message = completion.choices[0].message
                if message.tool_calls:
                    return {"type": "function_call",
                            "tool_calls": [
                                {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                for tc in message.tool_calls
                                ]
                            }
                return message.content or ""
            return ""


    def generate_embeddings(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        batch_size: int = 2048,
        **kwargs: Any
    ) -> List[List[float]]:
        effective_embedding_model = model_name or self.embedding_model_name
        all_embeddings: List[List[float]] = []

        processed_texts = [text.replace("\n", " ") for text in texts]

        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=effective_embedding_model
                )
                all_embeddings.extend([item.embedding for item in response.data])
            except APIError as e:
                print(f"OpenAI API Error in generate_embeddings for batch: {e}")
                raise RuntimeError(f"OpenAI API Error during embedding: {str(e)}") from e
            except Exception as e:
                print(f"An unexpected error occurred in OpenAIClient.generate_embeddings: {e}")
                raise RuntimeError(f"Unexpected error during embedding: {str(e)}") from e
        return all_embeddings

    def count_tokens(self, text: str, model_name: Optional[str] = None, **kwargs: Any) -> int:
        if not TIKTOKEN_AVAILABLE or tiktoken is None:
            print("Warning: tiktoken library is not available. Token counting will not be accurate. Returning len(text) / 4 as a rough estimate.")
            return len(text) // 4 # Rough estimate if tiktoken is missing

        effective_model_name = model_name or self.chat_model_name

        try:
            # Default to cl100k_base as it's used by modern OpenAI models
            encoding_name_to_try = "cl100k_base"
            try:
                # tiktoken.encoding_for_model() will raise KeyError if model not found
                encoding = tiktoken.encoding_for_model(effective_model_name)
            except KeyError:
                # Fallback for models not directly known by tiktoken but likely use cl100k_base
                print(f"Warning: Encoding for model '{effective_model_name}' not found by tiktoken.encoding_for_model(). Using '{encoding_name_to_try}' as fallback.")
                encoding = tiktoken.get_encoding(encoding_name_to_try)
        except Exception as e:
            print(f"Error getting tiktoken encoding for model {effective_model_name}: {e}. Using cl100k_base as a robust fallback.")
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))
