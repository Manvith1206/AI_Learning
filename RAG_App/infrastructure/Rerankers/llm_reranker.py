from .base_reranker import BaseReranker
import re
import time
import RAG_App.infrastructure.Common.RAG_Constants as constants
from RAG_App.infrastructure.LLM_Chat_Services.base_llm_service import BaseLLMService
class LLMReranker(BaseReranker):
    def __init__(self, llm_client: BaseLLMService, model="gemini-2.0-flash", top_k_for_reranking: int = 5):
        self.llm_client = llm_client
        self.model_name = model
        self.time_taken = 0
        self.cost = 0
        self.top_k_for_reranking = top_k_for_reranking
    def rerank(self, query, documents, **kwargs):
        
        chunk_list = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
        rerank_prompt = f"""
            Role: 
            Assume the role of a research assistant tasked with evaluating the relevance of 
            document chunks to a user query.

            Task: 
            You will receive a user query along with a list of retrieved document chunks. 
            Your objective is to assess the relevance of each chunk to the query.
            After your evaluation, you will rerank the chunks based on their relevance and provide a formal 
            explanation of your reasoning for the new ranking.

            Query: {query}

            Chunks:
            {chunk_list}

            Output Format:
            Please respond in the following format:
            Reranked Chunk(s): [list the chunk numbers]
            Explanation: [your reasoning for reranking the chunks]
            """
        start_time = time.time()
        full_response = ""
        for delta in self.llm_client.generate_response(rerank_prompt):
            full_response += delta

        print("topk", self.top_k_for_reranking)
        best_chunks_match = re.search(r"Reranked Chunk\(s\):\s*\[([^\]]+)\]", full_response)
        explanation_match = re.search(r"Explanation:\s*(.*)", full_response, re.DOTALL)
        selected_indices = []
        if best_chunks_match:
            indices_str = best_chunks_match.group(1)
            selected_indices = [int(idx.strip())-1 for idx in indices_str.split(",") if idx.strip().isdigit()]
        explanation = explanation_match.group(1).strip() if explanation_match else constants.NO_EXPLAINATION_NEEDED_MESSAGE
        selected_documents = [documents[i] for i in selected_indices if 0 <= i < len(documents)]
        print("SelectedDocs: ", selected_documents)
        print("Documents: ", documents)
        selected_documents = selected_documents[:self.top_k_for_reranking]
        if not selected_documents:
            selected_documents = documents
            explanation = constants.LLM_DID_NOT_SELECT_INFO_MESSAGE
        end_time = time.time()
        self.time_taken = end_time - start_time
        return selected_documents, explanation

    def get_cost_and_time_taken(self):
        """Returns the time taken for the rerank operation."""
        return self.cost, self.time_taken
