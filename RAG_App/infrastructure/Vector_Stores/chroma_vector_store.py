from .base_vector_store import BaseVectorStore
import chromadb
import numpy as np
import infrastructure.common.rag_constants as constants
import uuid
import time
from langchain_core.documents import Document as LangchainDocument

class ChromaVectorStore(BaseVectorStore):
    def __init__(self, collectionName, path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=path)
        self.collectionName = collectionName
        self.collection = self.client.get_or_create_collection(self.collectionName)
        self.documents = []
        self.time_taken = 0
        self.cost = 0
        self._load_all_documents()

    def _load_all_documents(self):
        """Loads all documents from the ChromaDB collection on initialization."""
        existing_docs = self.collection.get(include=["documents", "metadatas"])
        if existing_docs and existing_docs['ids']:
            self.documents = [
                LangchainDocument(
                    page_content=doc,
                    metadata=meta
                )
                for doc, meta in zip(existing_docs['documents'], existing_docs['metadatas'])
            ]

    def add_embeddings(self, embeddings, documents):
        start_time = time.time()
        
        self.documents.extend(documents)

        if hasattr(embeddings, "toarray"):
            emb_arr = embeddings.toarray().astype(np.float32).tolist()
        else:
            emb_arr = np.array(embeddings, dtype=np.float32).tolist()

        ids = [str(uuid.uuid4()) for _ in documents]
        
        metadatas = [doc.metadata if hasattr(doc, 'metadata') else {} for doc in documents]

        self.collection.add(
            ids=ids,
            embeddings=emb_arr,
            documents=[doc.page_content for doc in documents],
            metadatas=metadatas
        )
        
        end_time = time.time()
        self.time_taken = end_time - start_time

    def search(self, query_embedding, top_k=5):
        if hasattr(query_embedding, "toarray"):
            emb_list = query_embedding.toarray().astype(np.float32).tolist()
        else:
            emb_list = np.array(query_embedding, dtype=np.float32).tolist()

        results = self.collection.query(
            query_embeddings=emb_list, 
            n_results=top_k,
            include=["metadatas", "distances", "documents"]
        )
        
        formatted_results = []
        if results and results.get('ids') and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                score = results['distances'][0][i]
                doc = LangchainDocument(
                    page_content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i]
                )
                formatted_results.append({
                    constants.ID: results['ids'][0][i],
                    constants.Document: doc,
                    constants.Score: 1.0 - score if score is not None else 0.0
                })
        return formatted_results

    def format_documents(self, documents):
        return [doc.page_content for doc in documents]
    
    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken
    
    def update_index(self, index):
        pass
