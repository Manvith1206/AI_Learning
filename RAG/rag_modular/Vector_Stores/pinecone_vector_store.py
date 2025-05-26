from pinecone import Pinecone, ServerlessSpec
import os
from .base_vector_store import BaseVectorStore
import rag_modular.Common.RAG_Constants as constants
import uuid
import numpy as np
from scipy.sparse import csr_matrix

class PineConeVectorStore(BaseVectorStore):
    def __init__(self, api_key: str, index_name: str):
        self.documents = None
        self.embeddings = None
        self.index_name = index_name
        self.dimension = None
        self.pc = Pinecone(api_key=api_key)
        self.time_taken = 0
        self.cost = 0

    def add_embeddings(self, embeddings, documents):
        start_time = time.time()
        self.documents = documents
        self.embeddings = embeddings
        vectors = []
        
        # Connect to existing index or create one if needed
        if isinstance(embeddings, csr_matrix):
            self.dimension = embeddings.shape[1]
        if isinstance(embeddings, list):
            self.dimension = len(embeddings[0])

        if self.index_name not in [i.name for i in self.pc.list_indexes()] or self.dimension != [i.index['dimension'] for i in self.pc.list_indexes()]:
            if self.index_name in [i.name for i in self.pc.list_indexes()]:
                self.pc.delete_index(self.index_name)

            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,  # We'll set dimension at runtime
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        else:
            for i in self.pc.list_indexes():
                if i.index['dimension'] == self.dimension:
                    self.index_name = i.name
            
        
        self.index = self.pc.Index(self.index_name)

        # Handle sparse to dense if needed
        if hasattr(embeddings, "toarray"):
            embeddings = embeddings.toarray()

        # Ensure numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        # Convert to float32
        embeddings = embeddings.astype(np.float32)

        for i, (embedding, document) in enumerate(zip(embeddings, documents)):
            doc_id = document.get("id", str(uuid.uuid4()))
            metadata = {}

            # First, add the 'metadata' fields inside your document (flattened)
            if "metadata" in document and isinstance(document["metadata"], dict):
                for k, v in document["metadata"].items():
                    metadata[k] = v

            # Then add other fields like 'page_content'
            if "page_content" in document:
                metadata["page_content"] = document["page_content"]

            vectors.append({
                "id": doc_id,
                "values": embedding.tolist(),
                "metadata": metadata
            })

        # Upsert vectors into Pinecone
        self.index.upsert(vectors)
        end_time = time.time()
        self.time_taken = end_time - start_time

    def search(self, query_embedding, top_k=5):
        if self.index is None:
            raise ValueError("Vector store not initialized. Call add_embeddings first.")

        if hasattr(query_embedding, "toarray"):
            query_embedding = query_embedding.toarray()
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        query_vector = query_embedding[0].tolist()

        search_result = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        results = []
        
        for match in search_result["matches"]:
            metadata = match["metadata"]
            score = match["score"]
            id = match["id"]
            results.append({
                constants.Document: metadata,
                constants.Score: score,
                constants.ID: id
            })

        return results

    def format_documents(self, documents):
        return documents
        
    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken