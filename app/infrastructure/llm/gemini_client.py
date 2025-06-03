from typing import List, Dict, Any, Optional, Union, Iterator
from app.infrastructure.llm.base_llm import BaseLLM
from app.config import settings # To suggest where API keys might come from in instantiation phase

# Ensure google-generativeai is in requirements.txt
try:
    import google.generativeai as genai
    from google.generativeai.types import GenerateContentResponse, ContentDict, Tool, FunctionDeclaration
    from google.api_core.exceptions import GoogleAPIError
except ImportError:
    print("google-generativeai library not found. Please install it.")
    # Define dummy types for linting if library not present
    GenerateContentResponse = Any
    ContentDict = Dict
    Tool = Any
    FunctionDeclaration = Any
    GoogleAPIError = Exception
    genai = None # type: ignore


# Helper to map our message format to Gemini's
def _map_messages_to_gemini_format(messages: List[Dict[str, str]]) -> List[ContentDict]:
    gemini_messages: List[ContentDict] = []
    # Gemini expects roles to alternate starting with 'user'.
    # If history starts with 'assistant'/'model', it can cause issues.
    # This mapping assumes the input 'messages' list is already well-formed
    # in terms of conversation flow for the LLM.
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        # Gemini uses 'user' and 'model' roles
        gemini_role = "user" if role.lower() == "user" else "model"
        gemini_messages.append({'role': gemini_role, 'parts': [{'text': content}]})
    return gemini_messages

class GeminiClient(BaseLLM):
    def __init__(self,
                 api_key: str,
                 chat_model_name: str = "gemini-1.5-flash-latest", # More specific default
                 embedding_model_name: str = "models/text-embedding-004",
                 **kwargs: Any):
        super().__init__() # Call parent __init__
        if genai is None:
            raise ImportError("google-generativeai library is required to use GeminiClient.")

        # It's good practice to check if api_key is provided
        if not api_key:
            raise ValueError("Gemini API key must be provided.")

        genai.configure(api_key=api_key)
        self.chat_model_name = chat_model_name
        self.embedding_model_name = embedding_model_name

        # Initialize the generative model for chat
        # Safety measures for model initialization can be added if needed (e.g. list_models)
        self.chat_model = genai.GenerativeModel(self.chat_model_name)

        self.default_temperature = kwargs.get("temperature", 0.7)
        self.default_top_p = kwargs.get("top_p", 1.0) # Gemini often uses 1.0 for top_p
        self.default_top_k = kwargs.get("top_k", None) # Gemini uses top_k differently; sometimes not set or set to 1 or more. None is often fine.
        self.default_max_tokens = kwargs.get("max_output_tokens", 2048) # Example default

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs: Any
    ) -> Union[str, Iterator[str], Any]:

        gemini_formatted_messages = _map_messages_to_gemini_format(messages)

        generation_config_args = {
            "temperature": kwargs.get("temperature", self.default_temperature),
            "top_p": kwargs.get("top_p", self.default_top_p),
            "top_k": kwargs.get("top_k", self.default_top_k),
            "max_output_tokens": kwargs.get("max_output_tokens", self.default_max_tokens)
        }
        # Filter out None values for top_k, as API might expect it to be unset or an int
        if generation_config_args["top_k"] is None:
            del generation_config_args["top_k"]

        generation_config = genai.types.GenerationConfig(**generation_config_args) # type: ignore

        tools_arg: Optional[List[Tool]] = kwargs.get("tools")

        try:
            if stream and not tools_arg:
                response_iterator = self.chat_model.generate_content(
                    contents=gemini_formatted_messages,
                    generation_config=generation_config,
                    stream=True
                )
                def content_iterator():
                    for chunk in response_iterator:
                        # Check for parts and text within parts
                        if chunk.parts:
                            yield "".join(part.text for part in chunk.parts if hasattr(part, 'text') and part.text is not None)
                        # Check for text directly on the chunk (older API versions or simpler responses)
                        elif hasattr(chunk, 'text') and chunk.text is not None:
                            yield chunk.text
                return content_iterator()
            else:
                response: GenerateContentResponse = self.chat_model.generate_content(
                    contents=gemini_formatted_messages,
                    generation_config=generation_config,
                    tools=tools_arg if tools_arg else None # type: ignore
                )

                if response.candidates and response.candidates[0].content.parts:
                    # Check for function call
                    first_part = response.candidates[0].content.parts[0]
                    if hasattr(first_part, 'function_call') and first_part.function_call:
                        return {"type": "function_call",
                                "function_call": {
                                    "name": first_part.function_call.name,
                                    "args": dict(first_part.function_call.args) # Convert to dict
                                    }
                                }
                    # Standard text response
                    return "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text') and part.text is not None)
                # Fallback for simpler response structures or if parts are empty
                elif hasattr(response, 'text') and response.text is not None:
                    return response.text
                # If no text and no function call, might be a finish_reason like "SAFETY"
                elif response.candidates and response.candidates[0].finish_reason != 'STOP':
                     return f"Generation stopped: {response.candidates[0].finish_reason.name}"
                return ""

        except GoogleAPIError as e:
            print(f"Gemini API Error in chat: {e}")
            raise RuntimeError(f"Gemini API Error: {str(e)}") from e
        except Exception as e:
            print(f"An unexpected error occurred in GeminiClient.chat: {e}")
            raise RuntimeError(f"Unexpected error in Gemini chat: {str(e)}") from e

    def generate_embeddings(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
        title: Optional[str] = None,
        batch_size: int = 80, # Gemini API recommends max 100 per request for text-embedding-004
        **kwargs: Any
    ) -> List[List[float]]:

        effective_embedding_model = model_name or self.embedding_model_name
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                result = genai.embed_content(
                    model=effective_embedding_model,
                    content=batch_texts,
                    task_type=task_type.lower(), # API expects lowercase task_type
                    title=title if task_type.upper() == "RETRIEVAL_DOCUMENT" and title else None
                )
                # result['embedding'] should be List[List[float]]
                batch_embeddings = result.get('embedding')
                if isinstance(batch_embeddings, list):
                    all_embeddings.extend(batch_embeddings)
                else:
                    # Log an error or handle unexpected structure
                    print(f"Warning: Unexpected embedding structure for batch starting at index {i}. Result: {result}")

            except GoogleAPIError as e:
                print(f"Gemini API Error in generate_embeddings for batch: {e}")
                raise RuntimeError(f"Gemini API Error during embedding: {str(e)}") from e
            except Exception as e:
                print(f"An unexpected error occurred in GeminiClient.generate_embeddings: {e}")
                raise RuntimeError(f"Unexpected error during embedding: {str(e)}") from e
        return all_embeddings

    def count_tokens(self, text: str, model_name: Optional[str] = None, **kwargs: Any) -> int:
        effective_model_name = model_name or self.chat_model_name

        model_to_use_for_counting = self.chat_model
        if model_name and model_name != self.chat_model_name:
            # This might re-initialize a model just for counting tokens, which is okay.
            # Consider if genai offers a standalone tokenizer utility if performance is critical.
            model_to_use_for_counting = genai.GenerativeModel(effective_model_name) # Use effective_model_name

        try:
            # Count tokens for a single piece of text.
            # If 'text' is part of a larger message structure, ensure only text content is passed.
            token_count_response = model_to_use_for_counting.count_tokens(text)

            if isinstance(token_count_response, int):
                return token_count_response
            elif hasattr(token_count_response, 'total_tokens') and isinstance(token_count_response.total_tokens, int):
                return token_count_response.total_tokens
            else:
                print(f"Warning: count_tokens returned unexpected type: {type(token_count_response)}")
                return 0 # Fallback if structure is not as expected
        except GoogleAPIError as e:
            print(f"Gemini API Error in count_tokens: {e}")
            raise RuntimeError(f"Gemini API Error during token counting: {str(e)}") from e
        except Exception as e:
            print(f"An unexpected error occurred in GeminiClient.count_tokens: {e}")
            raise RuntimeError(f"Unexpected error during token counting: {str(e)}") from e
