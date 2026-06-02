from pinecone import Pinecone, ServerlessSpec
from .base_vector_store import BaseVectorStore
import rag_modular.RAG_Constants as constants
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
        

    def add_embeddings(self, embeddings, documents):
        self.documents = documents
        self.embeddings = embeddings
        vectors = []
        
        # Connect to existing index or create one if needed
        if isinstance(embeddings, csr_matrix):
            self.dimension = embeddings.shape[1]
        if isinstance(embeddings, list):
            self.dimension = len(embeddings[0])

        existing_indexes = list(self.pc.list_indexes())
        existing_names = [index.name for index in existing_indexes]
        existing_index = next((index for index in existing_indexes if index.name == self.index_name), None)
        existing_dimension = existing_index.dimension if existing_index else None

        if self.index_name not in existing_names or self.dimension != existing_dimension:
            if self.index_name in existing_names:
                self.pc.delete_index(self.index_name)

            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,  # We'll set dimension at runtime
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

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
