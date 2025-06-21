import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Union, Tuple
from .base_reranker import BaseReranker
import requests
import time
import infrastructure.common.rag_constants as constants
class JinaReranker(BaseReranker):
    """
    A reranker using JINA AI's reranker models from Hugging Face.
    """
    def __init__(self, api_key, model: str = "jinaai/jina-reranker-v1-base-en", top_k_for_reranking: int = 5):
        """
        Initialize the JINA reranker with a specified model.
        
        Args:
            model_name: The name of the JINA reranker model
            device: Device to run the model on ('cpu', 'cuda', etc.). If None, uses CUDA if available.
        """
        self.model = model
        self.time_taken = 0
        self.cost = 0
        self.top_k_for_reranking = top_k_for_reranking
        self.api_key = api_key

    def rerank(self, query, documents, **kwargs):
        import requests
        top_k = kwargs.get('top_k', 5)
        url = 'https://api.jina.ai/v1/rerank'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': self.api_key  # Ensure you have set this in your secrets
        }
        start_time = time.time()
        data = {
            "model": self.model,
            "query": query,
            "top_n": self.top_k_for_reranking,
            "documents": documents,
            "return_documents": False
        }

        response = requests.post(url, headers=headers, json=data)

         # Create a list of (document, score) tuples
        results = response.json().get('results', [])
        doc_score_pairs = [(documents[result['index']], result['relevance_score']) for result in results]

        # Sort by score in descending order (higher score = more relevant)
        sorted_results = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        # Extract just the sorted documents if needed
        sorted_documents = [doc for doc, score in sorted_results]
                            
        explaination = f"Jina Re ranking Model {self.model} re ranked the docs"
        
        end_time = time.time()
        self.time_taken = end_time - start_time
        return sorted_documents, explaination

    def get_cost_and_time_taken(self):
        """Returns the time taken for the rerank operation."""
        return self.cost, self.time_taken

