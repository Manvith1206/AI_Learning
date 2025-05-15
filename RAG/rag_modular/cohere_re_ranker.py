from .base_reranker import BaseReranker
import cohere as co
import RAG_Constants as constants

class CohereReranker(BaseReranker):
    def __init__(self, api_key: str = None, model_name: str = constants.CohereEmbedModels.COHERE_EMBED_MODEL_ENG):
        """
        api_key: your COHERE_API_KEY (or set via env var COHERE_API_KEY)
        model:   the Cohere embed model to use
        """
        
        key = api_key 
        if not key:
            raise ValueError("Cohere API key not provided. Set COHERE_API_KEY or pass api_key.")
        self.client = co.Client(key)
        self.model = model_name

    def rerank(self, query, documents, **kwargs):
        response = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model
        )
        
        # Create a list of (document, score) tuples
        doc_score_pairs = [(documents[result.index], result.relevance_score) for result in response.results]

        # Sort by score in descending order (higher score = more relevant)
        sorted_results = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        # Extract just the sorted documents if needed
        sorted_documents = [doc for doc, score in sorted_results]
                            
        explaination = f"Cohere Re ranking Model {self.model} re ranked the docs"
        return sorted_documents, explaination

