from pinecone import Pinecone, ServerlessSpec
from .base_vector_store import BaseVectorStore
import infrastructure.common.rag_constants as constants
import uuid
import numpy as np
import time
from langchain_core.documents import Document as LangchainDocument

class PineConeVectorStore(BaseVectorStore):
    def __init__(self, api_key: str, index_name: str, dimension: int = 1536):
        self.index_name = index_name
        self.dimension = dimension
        self.pc = Pinecone(api_key=api_key)
        self.time_taken = 0
        self.cost = 0
        self.index = None
        self.documents = []

        existing_indexes = [index_info.name for index_info in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        self.index = self.pc.Index(index_name)
        self._load_all_documents()

    def _load_all_documents(self):
        """Loads all documents from the Pinecone index using pagination."""
        try:
            all_ids = []
            next_token = None
            while True:
                list_response = self.index.list(limit=100, next_token=next_token)
                if 'vectors' in list_response and list_response['vectors']:
                    all_ids.extend([v['id'] for v in list_response['vectors']])
                next_token = list_response.get('pagination', {}).get('next')
                if not next_token:
                    break
            
            if not all_ids:
                return

            # Fetch vectors in batches
            for i in range(0, len(all_ids), 100):
                batch_ids = all_ids[i:i+100]
                fetch_response = self.index.fetch(ids=batch_ids)
                if 'vectors' in fetch_response:
                    for vec_id, vec_data in fetch_response['vectors'].items():
                        metadata = vec_data.get('metadata', {})
                        page_content = metadata.pop('page_content', '')
                        self.documents.append(
                            LangchainDocument(
                                page_content=page_content,
                                metadata=metadata
                            )
                        )
        except Exception as e:
            print(f"Could not load documents from Pinecone: {e}")
            pass

    def add_embeddings(self, embeddings, documents):
        start_time = time.time()
        
        if self.index is None:
            raise ValueError("Pinecone index not initialized.")

        self.documents.extend(documents)

        if hasattr(embeddings, "toarray"):
            embeddings = embeddings.toarray()
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        embeddings = embeddings.astype(np.float32)

        vectors_to_upsert = []
        for doc, embedding in zip(documents, embeddings):
            doc_id = str(uuid.uuid4())
            metadata = {**doc.metadata, 'page_content': doc.page_content}
            
            vectors_to_upsert.append({
                "id": doc_id,
                "values": embedding.tolist(),
                "metadata": metadata
            })

        if vectors_to_upsert:
            for i in range(0, len(vectors_to_upsert), 100):
                batch = vectors_to_upsert[i:i+100]
                self.index.upsert(vectors=batch)

        end_time = time.time()
        self.time_taken = end_time - start_time

    def search(self, query_embedding, top_k=5):
        if self.index is None:
            raise ValueError("Vector store not initialized.")

        if hasattr(query_embedding, "toarray"):
            query_embedding = query_embedding.toarray()
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32).tolist()

        search_result = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        results = []
        if search_result.get("matches"):
            for match in search_result["matches"]:
                metadata = match.get('metadata', {})
                page_content = metadata.pop('page_content', '')
                doc = LangchainDocument(
                    page_content=page_content,
                    metadata=metadata
                )
                results.append({
                    constants.Document: doc,
                    constants.Score: match["score"],
                    constants.ID: match["id"]
                })

        return results

    def format_documents(self, documents):
        return [doc.page_content for doc in documents]
        
    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken
    
    def update_index(self, index):
        pass
    
    def get_index(self):
        return None