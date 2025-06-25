import time
from .base_embedder import BaseEmbedder
from google import genai
import infrastructure.common.rag_constants as constants
from infrastructure.common.component_registry import register, EMBEDDERS_REGISTRY

@register(EMBEDDERS_REGISTRY, constants.EmbedderType.GEMINI.value)
class GeminiEmbedder(BaseEmbedder):
    def __init__(self, api_key=None, model_name = constants.GeminiEmbedModels.GEMINI_EMBED_001_MODEL.value):
        api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.model = model_name
        self.embeddings = []
        self.time_taken = 0
        self.cost = 0
    
    def batch_chunks(self, chunks, batch_size=80):
        """Yield successive batches of size batch_size."""
        
        for i in range(0, len(chunks), batch_size):
            yield chunks[i:i + batch_size]

    def embed_documents(self, texts):
        start_time = time.time() 
        all_new_embeddings_values = []  

        # Iterate over batches of texts.
        for text_batch in self.batch_chunks(texts, batch_size=80): 
            if not text_batch:
                continue

            try:
                resp = self.client.models.embed_content(
                    model=self.model,
                    contents=text_batch
                )
                
            except Exception as e:
                print(f"Error embedding batch: {e}")
                continue 
                
            current_batch_extracted_values = []
            if hasattr(resp, 'embeddings') and isinstance(resp.embeddings, list) and resp.embeddings:
                for embedding_structure in resp.embeddings:
                    current_batch_extracted_values = embedding_structure.embeddings

            all_new_embeddings_values.extend(current_batch_extracted_values)
        
        self.embeddings = all_new_embeddings_values
        end_time = time.time()
        self.time_taken = end_time - start_time

        return self.embeddings
    
    def transform(self, texts):
        start_time = time.time() 
        all_new_embeddings_values = [] 

        for text_batch in self.batch_chunks(texts, batch_size=80): 
            if not text_batch:  # Skip if the batch is empty
                continue

            try:
                resp = self.client.models.embed_content(
                    model=self.model,
                    contents=text_batch
                )
            except Exception as e:
                print(f"Error embedding batch: {e}") 
                continue 
                
            # Extract numerical embedding values from the response of the current batch
            current_batch_extracted_values = []
            if hasattr(resp, 'embeddings') and isinstance(resp.embeddings, list) and resp.embeddings:
                for embedding_structure in resp.embeddings:
                    current_batch_extracted_values = embedding_structure.embeddings

            all_new_embeddings_values.extend(current_batch_extracted_values)
        
        # self.embeddings should now store all the generated embeddings for the input texts
        self.embeddings = all_new_embeddings_values
        end_time = time.time()
        self.time_taken = end_time - start_time
        return self.embeddings
    
    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken