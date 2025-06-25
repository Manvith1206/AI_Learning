from .base_reranker import BaseReranker
import cohere as co
import time
import infrastructure.common.rag_constants as constants
from infrastructure.common.component_registry import register, RERANKERS_REGISTRY

@register(RERANKERS_REGISTRY, name=constants.RerankerType.COHERE.value)
class CohereReranker(BaseReranker):
    def __init__(self, api_key: str = None, model: str = constants.CohereEmbedModels.COHERE_EMBED_MODEL_ENG, top_k_for_reranking: int = 3):
        """
        api_key: your COHERE_API_KEY (or set via env var COHERE_API_KEY)
        model:   the Cohere embed model to use
        """
        
        key = api_key 
        if not key:
            raise ValueError("Cohere API key not provided. Set COHERE_API_KEY or pass api_key.")
        self.client = co.Client(key)
        self.model = model
        self.time_taken = 0
        self.cost = 0
        self.top_k_for_reranking = top_k_for_reranking

    def rerank(self, query, documents, **kwargs):
        start_time = time.time()
        response = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_n=self.top_k_for_reranking
        )
        # Create a list of (document, score) tuples
        doc_score_pairs = [(documents[result.index], result.relevance_score) for result in response.results]

        # Sort by score in descending order (higher score = more relevant)
        sorted_results = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        # Extract just the sorted documents if needed
        sorted_documents = [doc for doc, score in sorted_results]
                            
        explaination = f"Cohere Re ranking Model {self.model} re ranked the docs"
        current_cost_value = 0
        if response.meta and response.meta.billed_units and response.meta.billed_units.input_tokens is not None:
            current_cost_value += self.get_cost_based_on_model(response.meta.billed_units.input_tokens)
        else:
            print("Warning: Cohere API response did not include input_tokens. Cost metric might be inaccurate.")

        self.cost = current_cost_value
        end_time = time.time()
        self.time_taken = end_time - start_time

        return sorted_documents, explaination

    def get_cost_and_time_taken(self):
        """Returns the time taken for the rerank operation."""
        return self.cost, self.time_taken

    
    def get_cost_based_on_model(self, tokens):
        if self.model == constants.CohereRerankingModels.RERANK_DEFAULT_MODEL.value:
            return (tokens/1000) * 0.0001
        elif self.model == constants.CohereRerankingModels.RERANK_ENGLISH.value:
            return (tokens / 1000000) * 0.12
        elif self.model == constants.CohereRerankingModels.RERANK_MULTLINGUAL.value:
            pass