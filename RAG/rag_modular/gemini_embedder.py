from .base_embedder import BaseEmbedder
from google import genai
import os
import streamlit as st

class GeminiEmbedder(BaseEmbedder):
    def __init__(self, api_key=None):
        api_key = api_key or st.secrets["GEMINI_API_KEY"]
        self.client = genai.Client(api_key=api_key)
        self.model = "models/embedding-001"
        self.texts = None
        self.embeddings = None
    def fit(self, texts):
        self.texts = texts
        self.embeddings = [
            self.embeddings.get_embedding(
                model=self.model,
                text=text
            ).values for text in texts
        ]
        return self.embeddings
    def transform(self, texts):
        return [
            self.client.embeddings.get_embedding(
                model=self.model,
                text=text
            ).values for text in texts
        ]
