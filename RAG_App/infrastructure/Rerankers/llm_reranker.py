from .base_reranker import BaseReranker
import re
import time
import infrastructure.common.rag_constants as constants
from infrastructure.llm_chat_services.base_llm_service import BaseLLMService
from infrastructure.prompt_providers.llm_reranker_prompt_provider import LLM_Reranker_Prompt_Provider
from infrastructure.common.component_registry import register, RERANKERS_REGISTRY

@register(RERANKERS_REGISTRY, name=constants.RerankerType.LLM.value)
class LLMReranker(BaseReranker):
    def __init__(self, top_k_for_reranking: int = 5):
        self.time_taken = 0
        self.cost = 0
        self.top_k_for_reranking = top_k_for_reranking
        
    def rerank(self, query, documents, llm_client: BaseLLMService, **kwargs):
        llm_reranker_prompt_provider = LLM_Reranker_Prompt_Provider()
        chunk_list = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
        rerank_prompt = llm_reranker_prompt_provider.get_final_prompt(query=query, chunk_list=chunk_list)
        start_time = time.time()
        full_response = ""
        for delta in llm_client.generate_response(rerank_prompt):
            full_response += delta

        best_chunks_match = re.search(r"Reranked Chunk\(s\):\s*\[([^\]]+)\]", full_response)
        explanation_match = re.search(r"Explanation:\s*(.*)", full_response, re.DOTALL)
        selected_indices = []
        if best_chunks_match:
            indices_str = best_chunks_match.group(1)
            selected_indices = [int(idx.strip())-1 for idx in indices_str.split(",") if idx.strip().isdigit()]
        explanation = explanation_match.group(1).strip() if explanation_match else constants.UIDisplayNameConstants.NO_EXPLAINATION_NEEDED_MESSAGE
        selected_documents = [documents[i] for i in selected_indices if 0 <= i < len(documents)]
        
        selected_documents = selected_documents[:self.top_k_for_reranking]
        if not selected_documents:
            selected_documents = documents
            explanation = constants.UIDisplayNameConstants.LLM_DID_NOT_SELECT_INFO_MESSAGE
        end_time = time.time()
        self.time_taken = end_time - start_time
        return selected_documents, explanation

    def get_cost_and_time_taken(self):
        """Returns the time taken for the rerank operation."""
        return self.cost, self.time_taken
