from .base_vector_store import BaseVectorStore
import chromadb
import numpy as np
import infrastructure.common.RAG_Constants as constants
import time
from infrastructure.common.component_registry import register, VECTOR_STORES_REGISTRY

@register(VECTOR_STORES_REGISTRY, name=constants.VectorStore.CHROMA.value)
class ChromaVectorStore(BaseVectorStore):
    def __init__(self, collectionName):
        self.client = chromadb.PersistentClient()
        self.collectionName = collectionName
        self.documents = []
        self.time_taken = 0
        self.cost = 0

    def update_index(self, index):
        self.collection = self.client.get_collection(index)
        existing_docs = self.collection.get(include=["documents", "metadatas"])
        self.documents.clear()
        for idx, (doc, meta, ids) in enumerate(zip(existing_docs['documents'], existing_docs['metadatas'], existing_docs['ids']), start=1):
            (f"ID: {idx}, page_content: {doc}, metadata: {meta}")
            self.documents.append({
                constants.Constants.ID: ids,
                constants.Constants.PAGE_CONTENT: doc,
                constants.Constants.METADATA: meta
            })
        
    def add_embeddings(self, embeddings, documents):
        """
        Add embeddings to the vector store.
        
        Args:
            embeddings: The embeddings to add.
            documents: The documents associated with the embeddings.
        """
        # Implement logic to add embeddings to Chroma vector store
        start_time = time.time()
        self.embeddings = embeddings
        self.documents = documents

        # for i in range(0, self.client.list_collections().count()):
        #     if self.client.list_collections()[i].name == self.collectionName:
        #         self.client.delete_collection(collectionName)

        if hasattr(embeddings, "toarray"):
            emb_arr = self.embeddings.toarray().astype(np.float32)
        else:
            emb_arr = np.array(self.embeddings, dtype=np.float32)

        collection = self.client.get_or_create_collection(self.collectionName)
        self.collection = collection
        ids = [f"doc_{i}" for i in range(len(self.documents))]  # Auto-generate IDs
        collection.upsert(
            ids=ids,
            embeddings=emb_arr,
            documents=[doc[constants.Constants.PAGE_CONTENT] for doc in self.documents]
        )
        end_time = time.time()
        self.time_taken = end_time - start_time

    def search(self, query_embedding, top_k=5):
        """
        Search for most similar documents in Chroma vector store.   
        """
        if hasattr(query_embedding, "toarray"):
            emb_arr = query_embedding.toarray().astype(np.float32)
        else:
            emb_arr = np.array(query_embedding, dtype=np.float32)

        results = self.collection.query(query_embeddings=emb_arr, 
                                        n_results=top_k)
        
        ids = results["ids"]
        docs = results["documents"][0]
        distances = np.array(results['distances'][0]).flatten()
        normalized = (distances - distances.min()) / (distances.max() - distances.min())

        similarity_scores = 1 - normalized  # if distance is in [0, 1]
        
        for id, doc, score in zip(ids, docs, similarity_scores):
            print(f"Format ID: {id}, Document: {doc}, Score: {score}")
        formatted_results = [
            {constants.Constants.ID: id_, constants.Constants.Document: {constants.Constants.PAGE_CONTENT: doc}, constants.Constants.Score: float(score)}
            for id_, doc, score in zip(ids, docs, similarity_scores)
        ]
        print("Formatted Results", formatted_results)

        return formatted_results

    def get_all_documents(self):
        return self.documents
    
    def format_documents(self, documents):
        return documents
    
    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken
    
    def get_index(self):
        return self.collectionName