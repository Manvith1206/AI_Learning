import voyageai
import voyageai.client
from .base_embedder import BaseEmbedder

class VoyageEmbedder(BaseEmbedder):
    def __init__(self, api_key, model):
        """
        api_key: your COHERE_API_KEY (or set via env var COHERE_API_KEY)
        model:   the Cohere embed model to use
        """ 
        key = api_key
        self.client = voyageai.Client(api_key=key)
        self.model = model
        self.embeddings = None
        
    def fit(self, texts):
        """
        For Voyage embeddings we don't need a separate fit step;
        we just embed the texts and cache if desired.
        """ 
        emb = self.client.embed(texts, self.model)
        self.embeddings = emb.embeddings

        return self.embeddings
    def transform(self, texts):
        """
        Embed new Texts on demand
        """ 
        resp = self.client.embed(texts=texts, model=self.model)
        return resp.embeddings
