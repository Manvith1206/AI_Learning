import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Union, Tuple
from .base_reranker import BaseReranker
import requests
import streamlit as st
import rag_modular.Common.RAG_Constants as constants

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
        self.model = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    
    def rerank(self, query, documents, **kwargs):
        import requests
        top_k = kwargs.get('top_k', 5)
        url = 'https://api.jina.ai/v1/rerank'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': st.secrets[constants.JINA_RERANKER_API_KEY]
        }
        data = {
            "model": self.model,
            "query": query,
            "top_n": top_k,
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
        
        return sorted_documents, explaination


