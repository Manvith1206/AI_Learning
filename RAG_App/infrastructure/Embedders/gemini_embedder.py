import time
from .base_embedder import BaseEmbedder
from google import genai
import os
import streamlit as st
import RAG_App.infrastructure.Common.RAG_Constants as constants

class GeminiEmbedder(BaseEmbedder):
    def __init__(self, api_key=None, model_name = constants.GeminiEmbedModels.GEMINI_EMBED_001_MODEL.value):
        api_key = api_key or st.secrets[constants.GEMINI_API_KEY]
        self.client = genai.Client(api_key=api_key)
        self.model = model_name
        self.texts = None
        self.embeddings = []
        self.time_taken = 0
        self.cost = 0
    
    def batch_chunks(self, chunks, batch_size=80):
        """Yield successive batches of size batch_size."""
        
        for i in range(0, len(chunks), batch_size):
            yield chunks[i:i + batch_size]

    def fit(self, texts):
        start_time = time.time()  # Start timing the embedding process
        self.texts = texts  # Store original texts if needed for other purposes

        all_new_embeddings_values = []  # To store embeddings from all batches

        # Iterate over batches of texts.
        # self.batch_chunks should yield/return a list of strings for each batch.
        # Example: batch_size=80 means each text_batch will have up to 80 strings.
        for text_batch in self.batch_chunks(texts, batch_size=80): 
            if not text_batch:  # Skip if the batch is empty
                continue

            # Get embeddings for the current batch
            try:
                resp = self.client.models.embed_content(
                    model=self.model,
                    contents=text_batch  # Pass the individual batch (list of strings)
                )
            except Exception as e:
                # Consider logging the error and deciding how to handle failed batches
                # For example, you could skip this batch or raise the exception.
                print(f"Error embedding batch: {e}") 
                # If you need to maintain a 1:1 correspondence between input texts and embeddings,
                # you might add placeholders for embeddings of texts in a failed batch.
                continue 
                
            # Extract numerical embedding values from the response of the current batch
            current_batch_extracted_values = []
            if hasattr(resp, 'embeddings') and isinstance(resp.embeddings, list) and resp.embeddings:
                # resp.embeddings is a list of embedding structures (e.g., EmbeddingDict or Embedding objects).
                # Each structure should correspond to one text string in the input text_batch.
                for embedding_structure in resp.embeddings:
                    if hasattr(embedding_structure, "values") and embedding_structure.values is not None:
                        current_batch_extracted_values.append(embedding_structure.values)
                    # Handle cases where embeddings might be in a dictionary (e.g. google.generativeai.types.EmbeddingDict)
                    elif isinstance(embedding_structure, dict) and "values" in embedding_structure and embedding_structure["values"] is not None:
                        current_batch_extracted_values.append(embedding_structure["values"])
                    elif hasattr(embedding_structure, "embedding") and embedding_structure.embedding is not None: # Fallback for other possible structures
                        current_batch_extracted_values.append(embedding_structure.embedding)
                    else:
                        # Log or handle cases where an embedding structure is not as expected
                        print(f"Warning: Could not extract embedding values from: {embedding_structure}")
                        # Optionally, add a placeholder (e.g., None or a zero vector) if strict
                        # correspondence is needed and an embedding is missing.

            all_new_embeddings_values.extend(current_batch_extracted_values)
        
        # self.embeddings should now store all the generated embeddings for the input texts
        self.embeddings = all_new_embeddings_values
        end_time = time.time()  # End timing the embedding process
        self.time_taken = end_time - start_time

        return self.embeddings
    
    def transform(self, texts):
        start_time = time.time()  # Start timing the embedding process
        all_new_embeddings_values = []  # To store embeddings from all batches

        # Iterate over batches of texts.
        # self.batch_chunks should yield/return a list of strings for each batch.
        # Example: batch_size=80 means each text_batch will have up to 80 strings.
        for text_batch in self.batch_chunks(texts, batch_size=80): 
            if not text_batch:  # Skip if the batch is empty
                continue

            # Get embeddings for the current batch
            try:
                resp = self.client.models.embed_content(
                    model=self.model,
                    contents=text_batch  # Pass the individual batch (list of strings)
                )
            except Exception as e:
                # Consider logging the error and deciding how to handle failed batches
                # For example, you could skip this batch or raise the exception.
                print(f"Error embedding batch: {e}") 
                # If you need to maintain a 1:1 correspondence between input texts and embeddings,
                # you might add placeholders for embeddings of texts in a failed batch.
                continue 
                
            # Extract numerical embedding values from the response of the current batch
            current_batch_extracted_values = []
            if hasattr(resp, 'embeddings') and isinstance(resp.embeddings, list) and resp.embeddings:
                # resp.embeddings is a list of embedding structures (e.g., EmbeddingDict or Embedding objects).
                # Each structure should correspond to one text string in the input text_batch.
                for embedding_structure in resp.embeddings:
                    if hasattr(embedding_structure, "values") and embedding_structure.values is not None:
                        current_batch_extracted_values.append(embedding_structure.values)
                    # Handle cases where embeddings might be in a dictionary (e.g. google.generativeai.types.EmbeddingDict)
                    elif isinstance(embedding_structure, dict) and "values" in embedding_structure and embedding_structure["values"] is not None:
                        current_batch_extracted_values.append(embedding_structure["values"])
                    elif hasattr(embedding_structure, "embedding") and embedding_structure.embedding is not None: # Fallback for other possible structures
                        current_batch_extracted_values.append(embedding_structure.embedding)
                    else:
                        # Log or handle cases where an embedding structure is not as expected
                        print(f"Warning: Could not extract embedding values from: {embedding_structure}")
                        # Optionally, add a placeholder (e.g., None or a zero vector) if strict
                        # correspondence is needed and an embedding is missing.

            all_new_embeddings_values.extend(current_batch_extracted_values)
        
        # self.embeddings should now store all the generated embeddings for the input texts
        self.embeddings = all_new_embeddings_values
        end_time = time.time()
        self.time_taken = end_time - start_time
        return self.embeddings
    
    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken