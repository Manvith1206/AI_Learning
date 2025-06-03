from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseLLM(ABC):
    """
    Abstract base class for Large Language Model interactions.
    Defines a common interface for various LLM providers.
    """

    @abstractmethod
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, **kwargs: Any):
        """
        Initialize the LLM client.
        Args:
            api_key (Optional[str]): The API key for the LLM service.
            model_name (Optional[str]): The specific model to use.
            **kwargs: Additional provider-specific arguments.
        """
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs: Any
    ) -> Any: # Could be a string, or a generator for streaming
        """
        Send a chat request to the LLM.
        Args:
            messages (List[Dict[str, str]]): A list of message objects (e.g., [{'role': 'user', 'content': 'Hello'}]).
            stream (bool): Whether to stream the response.
            **kwargs: Additional provider-specific arguments for the chat call.
        Returns:
            Any: The LLM's response, or a generator if streaming.
        """
        pass

    @abstractmethod
    def generate_embeddings(
        self,
        texts: List[str],
        **kwargs: Any
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        Args:
            texts (List[str]): A list of texts to embed.
            **kwargs: Additional provider-specific arguments for embedding generation.
        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str, **kwargs: Any) -> int:
        """
        Count the number of tokens in a given text according to the model's tokenizer.
        Args:
            text (str): The text to tokenize.
            **kwargs: Additional provider-specific arguments.
        Returns:
            int: The number of tokens.
        """
        pass

    # You can add other common methods like:
    # - fine_tuning
    # - model_management (list_models, etc.)
    # - specific function calling setup if it can be generalized
