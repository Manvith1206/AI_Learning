import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Union, Tuple
from .base_reranker import BaseReranker
import requests

class JinaReranker(BaseReranker):
    """
    A reranker using JINA AI's reranker models from Hugging Face.
    """
    def __init__(self, model_name: str = "jinaai/jina-reranker-v1-base-en", device: str = None):
        """
        Initialize the JINA reranker with a specified model.
        
        Args:
            model_name: The name of the JINA reranker model
            device: Device to run the model on ('cpu', 'cuda', etc.). If None, uses CUDA if available.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    
    def rerank(self, query, documents, **kwargs):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer jina_65da3b25f83444649e9b746b986b4a03hzkxK8XzGFx6uJBXsFtlm2li_ADD"
        }

        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": 3,
            "documents": [documents],
            "return_documents": False
        }

        response = requests.post('https://api.jina.ai/v1/rerank', headers=headers, json=data)

        print("Response")
        print(response.json())
