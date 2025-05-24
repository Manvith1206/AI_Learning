from .base_embedder import BaseEmbedder
from google import genai
import os
import streamlit as st
import rag_modular.Common.RAG_Constants as constants

class GeminiEmbedder(BaseEmbedder):
    def __init__(self, api_key=None, model_name = constants.GeminiEmbedModels.GEMINI_EMBED_001_MODEL.value):
        api_key = api_key or st.secrets[constants.GEMINI_API_KEY]
        self.client = genai.Client(api_key=api_key)
        self.model = model_name
        self.texts = None
        self.embeddings = []
    
    def batch_chunks(self, chunks, batch_size=80):
        """Yield successive batches of size batch_size."""
        
        for i in range(0, len(chunks), batch_size):
            yield chunks[i:i + batch_size]

    def fit(self, texts):
        self.texts = texts
        batched_chunks = self.batch_chunks(texts, batch_size=80)
        resp = self.client.models.embed_content(
            model=self.model,
            contents=batched_chunks
        )
        
        if isinstance(resp.embeddings, list) and resp.embeddings:
            first = resp.embeddings[0]
            if hasattr(first, "values"):
                resp.embeddings = [e.values for e in resp.embeddings]
            elif hasattr(first, "embedding"):
                resp.embeddings = [e.embedding for e in resp.embeddings]
        self.embeddings.extend(resp.embeddings)
        
        return self.embeddings
    
    def transform(self, texts):
        resp = self.client.models.embed_content(
            model=self.model,
            contents=texts
        )
        if isinstance(resp.embeddings, list) and resp.embeddings:
            first = resp.embeddings[0]
            if hasattr(first, "values"):
                resp.embeddings = [e.values for e in resp.embeddings]
            elif hasattr(first, "embedding"):
                resp.embeddings = [e.embedding for e in resp.embeddings]
        self.embeddings = resp.embeddings

        return self.embeddings