from .base_embedder import BaseEmbedder
from google import genai
import streamlit as st
import rag_modular.RAG_Constants as constants

class GeminiEmbedder(BaseEmbedder):
    def __init__(self, api_key=None, model_name = constants.GeminiEmbedModels.GEMINI_EMBED_001_MODEL.value):
        api_key = api_key or st.secrets[constants.GEMINI_API_KEY]
        self.client = genai.Client(api_key=api_key)
        self.model = model_name
        self.texts = None
        self.embeddings = None
        
    def fit(self, texts):
        self.texts = texts
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
