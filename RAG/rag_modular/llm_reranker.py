from .base_reranker import BaseReranker
import re
import rag_modular.RAG_Constants as constants

class LLMReranker(BaseReranker):
    def __init__(self, llm_client, model_name="gemini-2.0-flash"):
        self.llm_client = llm_client
        self.model_name = model_name
    def rerank(self, query, documents, **kwargs):
        
        chunk_list = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
        rerank_prompt = f"""
            You are given a user query and a list of retrieved document chunks. 
            For each chunk, determine how relevant it is to the query, then select the best chunk(s) that would help answer the query. 
            Explain your reasoning for selecting the chunk(s).

            Query: {query}

            Chunks:
            {chunk_list}

            Please respond in the following format:
            Best Chunk(s): [list the chunk numbers]
            Explanation: [your reasoning]
            """
        response = self.llm_client.generate_response(
            prompt=rerank_prompt
        )
        response_text = response.strip()
        best_chunks_match = re.search(r"Best Chunk\(s\):\s*\[([^\]]+)\]", response_text)
        explanation_match = re.search(r"Explanation:\s*(.*)", response_text, re.DOTALL)
        selected_indices = []
        if best_chunks_match:
            indices_str = best_chunks_match.group(1)
            selected_indices = [int(idx.strip())-1 for idx in indices_str.split(",") if idx.strip().isdigit()]
        explanation = explanation_match.group(1).strip() if explanation_match else constants.NO_EXPLAINATION_NEEDED_MESSAGE
        selected_documents = [documents[i] for i in selected_indices if 0 <= i < len(documents)]
        if not selected_documents:
            selected_documents = documents
            explanation = constants.LLM_DID_NOT_SELECT_INFO_MESSAGE
        return selected_documents, explanation
