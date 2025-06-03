from .base_vector_store import BaseVectorStore
import chromadb
import numpy as np
import RAG_App.infrastructure.Common.RAG_Constants as constants

class ChromaVectorStore(BaseVectorStore):
    def __init__(self, collectionName):
        self.client = chromadb.PersistentClient()
        self.collectionName = collectionName
        
    def add_embeddings(self, embeddings, documents):
        """
        Add embeddings to the vector store.
        
        Args:
            embeddings: The embeddings to add.
            documents: The documents associated with the embeddings.
        """
        # Implement logic to add embeddings to Chroma vector store
        self.embeddings = embeddings
        self.documents = documents

        # for i in range(0, self.client.list_collections().count()):
        #     if self.client.list_collections()[i].name == self.collectionName:
        #         self.client.delete_collection(collectionName)
        print("First", type(self.embeddings))

        if hasattr(embeddings, "toarray"):
            emb_arr = self.embeddings.toarray().astype(np.float32)
        else:
            emb_arr = np.array(self.embeddings, dtype=np.float32)

        print("Second", type(emb_arr))
        collection = self.client.get_or_create_collection(self.collectionName)
        self.collection = collection
        ids = [f"doc_{i}" for i in range(len(self.documents))]  # Auto-generate IDs
        collection.upsert(
            ids=ids,
            embeddings=emb_arr,
            documents=self.documents
        )
        
    
        pass
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
        print("Similarity Scores", similarity_scores)
        print("IDs", ids)
        print("Documents", docs)
        print("Type of Docs", type(docs))
        for id, doc, score in zip(ids, docs, similarity_scores):
            print(f"Format ID: {id}, Document: {doc}, Score: {score}")
        formatted_results = [
            {constants.ID: id_, constants.Document: {constants.PAGE_CONTENT: doc}, constants.Score: float(score)}
            for id_, doc, score in zip(ids, docs, similarity_scores)
        ]
        print("Formatted Results", formatted_results)

        return formatted_results

    def format_documents(self, documents):
        formatted_documents = []
        for doc in documents:
            # Assuming each document is a dictionary with 'id' and 'text' keys
            formatted_doc = doc[constants.PAGE_CONTENT]

            formatted_documents.append(formatted_doc)

        return formatted_documents
    
    def get_cost_and_time_taken(self):
        pass
