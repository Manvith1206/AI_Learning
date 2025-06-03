from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np # Ensure numpy is here for CosineReranker at least

# from app.config import settings # For API keys
# For LLM/Embedder clients that might be needed by rerankers
from app.infrastructure.llm.base_llm import BaseLLM

# Ensure necessary libraries are in requirements.txt: cohere, transformers, sentence-transformers, torch
try:
    import cohere as cohere_client
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    cohere_client = None

try:
    # For Jina, which typically uses transformers
    # from transformers import AutoModelForSequenceClassification, AutoTokenizer
    # For now, JinaReranker uses requests to an API endpoint
    JINA_TRANSFORMERS_AVAILABLE = True # Placeholder, actual check might be more specific
except ImportError:
    JINA_TRANSFORMERS_AVAILABLE = False

try:
    # For CosineReranker if it uses sentence-transformers via an injected embedder
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True # For type hinting if needed
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class BaseReranker(ABC):
    def __init__(self):
        self.time_taken: float = 0.0
        self.cost: float = 0.0

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]], # Expects list of dicts with 'page_content', 'id', 'metadata'
        top_k: int,
        **kwargs: Any
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]: # Returns reranked documents and an optional explanation
        pass

    def get_cost_and_time_taken(self) -> tuple[float, float]:
        return self.cost, self.time_taken

class CohereReranker(BaseReranker):
    def __init__(self, api_key: str, model: str = "rerank-english-v2.0"):
        super().__init__()
        if not COHERE_AVAILABLE:
            raise ImportError("cohere library is required for CohereReranker.")
        if not api_key:
            raise ValueError("Cohere API key must be provided.")
        self.client = cohere_client.Client(api_key)
        self.model = model

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int,
        **kwargs: Any
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        start_time = time.time()

        doc_texts = [doc.get("page_content", "") for doc in documents]

        try:
            response = self.client.rerank(
                query=query,
                documents=doc_texts,
                model=self.model,
                top_n=top_k
            )
        except Exception as e:
            print(f"Cohere API call failed: {e}")
            # Fallback: return original documents if API fails
            return documents[:top_k], f"Cohere reranking failed: {e}"

        reranked_results_from_cohere = response.results

        # Map Cohere results back to original document dicts
        output_documents: List[Dict[str, Any]] = []
        for cohere_res in reranked_results_from_cohere:
            original_doc_index = cohere_res.index
            if 0 <= original_doc_index < len(documents):
                # Add/update relevance score to the original document dictionary
                doc_to_add = documents[original_doc_index].copy() # Avoid modifying original list items
                doc_to_add["rerank_score"] = cohere_res.relevance_score
                output_documents.append(doc_to_add)

        explanation = f"Reranked using Cohere model {self.model}."
        self.time_taken = time.time() - start_time
        return output_documents, explanation

class CosineReranker(BaseReranker):
    def __init__(self, embedding_client: BaseLLM): # Needs an embedder
        super().__init__()
        self.embedding_client = embedding_client
        # sklearn.metrics.pairwise.cosine_similarity will be used if numpy is available
        # Ensure numpy is in requirements for this to work smoothly if not using sklearn directly
        if not hasattr(np, "array"): # Basic check for numpy
             raise ImportError("numpy is required for CosineReranker's similarity calculations if not using sklearn.")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int,
        **kwargs: Any
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        start_time = time.time()

        doc_texts = [doc.get("page_content", "") for doc in documents]
        if not doc_texts:
            return [], "No documents to rerank."

        try:
            query_vec = self.embedding_client.generate_embeddings(texts=[query])[0]
            doc_vecs = self.embedding_client.generate_embeddings(texts=doc_texts)
        except Exception as e:
            print(f"Error generating embeddings for CosineReranker: {e}")
            return documents[:top_k], f"Embedding generation failed for CosineReranker: {e}"

        # Compute cosine similarities (assuming query_vec is 1D and doc_vecs is 2D)
        # query_vec needs to be (1, dim) for sklearn's cosine_similarity with (N, dim)
        query_vec_np = np.array(query_vec).reshape(1, -1)
        doc_vecs_np = np.array(doc_vecs)

        if query_vec_np.shape[1] != doc_vecs_np.shape[1]:
             return documents[:top_k], "Query and document embedding dimensions mismatch."

        sims = np.dot(doc_vecs_np, query_vec_np.T).flatten() / (np.linalg.norm(doc_vecs_np, axis=1) * np.linalg.norm(query_vec_np))

        # Pair docs with scores
        # Need to associate scores with original document dicts
        scored_documents = []
        for i, doc_dict in enumerate(documents):
            new_doc_dict = doc_dict.copy()
            new_doc_dict["rerank_score"] = sims[i]
            scored_documents.append(new_doc_dict)

        # Sort by new rerank_score
        scored_documents.sort(key=lambda x: x["rerank_score"], reverse=True)

        explanation = "Reranked by cosine similarity between query and document embeddings."
        self.time_taken = time.time() - start_time
        return scored_documents[:top_k], explanation


class JinaReranker(BaseReranker):
    # This implementation uses Jina's API endpoint directly via requests
    # Ensure 'requests' is in requirements.txt
    def __init__(self, api_key: str, model: str = "jina-reranker-v1-base-en"):
        super().__init__()
        if not api_key:
            raise ValueError("Jina API key must be provided.")
        self.api_key = api_key
        self.model = model
        self.api_url = 'https://api.jina.ai/v1/rerank'

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int,
        **kwargs: Any
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        import requests # Local import to ensure it's only needed if this class is used
        import re # Local import for LLMReranker, but good to have at top for rerankers.py
        start_time = time.time()

        doc_texts = [doc.get("page_content", "") for doc in documents]

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}' # Corrected Authorization format
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": doc_texts,
            "top_n": top_k,
            "return_documents": False # We only need scores and indices
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            api_results = response.json().get('results', [])
        except requests.exceptions.RequestException as e:
            print(f"Jina API call failed: {e}")
            return documents[:top_k], f"Jina API reranking failed: {e}"

        output_documents: List[Dict[str, Any]] = []
        for api_res in api_results:
            original_doc_index = api_res['index']
            if 0 <= original_doc_index < len(documents):
                doc_to_add = documents[original_doc_index].copy()
                doc_to_add["rerank_score"] = api_res['relevance_score']
                output_documents.append(doc_to_add)

        explanation = f"Reranked using Jina API model {self.model}."
        self.time_taken = time.time() - start_time
        return output_documents, explanation


class LLMReranker(BaseReranker):
    def __init__(self, llm_client: BaseLLM, model_name: Optional[str] = None): # Takes a BaseLLM client
        super().__init__()
        self.llm_client = llm_client
        self.model_name = model_name # Optional: model can be specified on llm_client or here

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int,
        **kwargs: Any
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        import re # Ensure re is available
        start_time = time.time()

        if not documents:
            return [], "No documents to rerank."

        # Prepare document strings for the prompt
        chunk_list_str = ""
        for i, doc_dict in enumerate(documents):
            # Using original index + 1 for human-readable numbering in prompt
            chunk_list_str += f"{i+1}. {doc_dict.get('page_content', '')}\n"

        # Construct prompt (simplified from original for clarity)
        # This prompt asks for indices, which can be error-prone for LLMs.
        # A more robust approach might involve asking for scores or pairwise comparisons.
        rerank_prompt = (
            f"Query: {query}\n\n"
            f"Document Chunks:\n{chunk_list_str}\n"
            f"Based on the query, which of the above document chunks (by number) are most relevant? "
            f"List the numbers of the top {min(top_k, len(documents))} most relevant chunks, comma-separated. "
            f"Example: 1, 3, 2"
        )

        messages = [{"role": "user", "content": rerank_prompt}]

        try:
            # Assuming llm_client.chat returns a response object with a 'content' attribute or similar
            # Or is a string directly
            response_data = self.llm_client.chat(messages=messages, model_name=self.model_name) # Pass model if needed by client

            llm_response_text = ""
            if isinstance(response_data, str):
                llm_response_text = response_data
            elif hasattr(response_data, 'content'): # Example for some clients
                llm_response_text = response_data.content
            elif isinstance(response_data, dict) and 'choices' in response_data: # OpenAI like
                 llm_response_text = response_data['choices'][0]['message']['content']
            else: # Fallback or handle other structures
                llm_response_text = str(response_data)

        except Exception as e:
            print(f"LLM call for reranking failed: {e}")
            return documents[:top_k], f"LLM reranking failed: {e}"

        # Parse LLM response (this part is fragile and LLM-dependent)
        selected_indices_from_llm: List[int] = []
        try:
            # Attempt to extract comma-separated numbers
            # Regex to find numbers, robust to some LLM formatting quirks
            found_numbers = re.findall(r'\d+', llm_response_text)
            selected_indices_from_llm = [int(num_str) - 1 for num_str in found_numbers] # Adjust to 0-based index
        except ValueError:
            print(f"Could not parse indices from LLM response: '{llm_response_text}'")
            # Fallback or error handling

        output_documents: List[Dict[str, Any]] = []
        seen_indices = set()
        for original_idx in selected_indices_from_llm:
            if 0 <= original_idx < len(documents) and original_idx not in seen_indices:
                output_documents.append(documents[original_idx])
                seen_indices.add(original_idx)

        # If LLM fails to select or selects too few, fill with original documents
        if len(output_documents) < top_k:
            for doc_dict in documents:
                if len(output_documents) >= top_k:
                    break
                # Check if doc_dict (by its original index or ID) is already included
                # This requires knowing original index or having unique IDs.
                # For simplicity here, we'll just add if not an exact object match (shallow).
                # A robust way would be to track original indices of documents if they are not already unique.
                is_already_added = any(added_doc is doc_dict for added_doc in output_documents) # Basic check
                if not is_already_added:
                     output_documents.append(doc_dict)


        explanation = f"Reranked by LLM. Raw response: {llm_response_text[:100]}..."
        self.time_taken = time.time() - start_time
        return output_documents[:top_k], explanation


def get_reranker(
    reranker_type: str,
    params: Optional[Dict[str, Any]] = None,
    # Explicitly pass clients if they are needed by specific rerankers
    # This avoids service locator pattern inside get_reranker
    llm_client_for_reranker: Optional[BaseLLM] = None,
    embedding_client_for_reranker: Optional[BaseLLM] = None,
    api_key_for_reranker: Optional[str] = None # For Cohere/Jina
    ) -> BaseReranker:

    params = params or {}

    if reranker_type == "cohere":
        api_key = api_key_for_reranker # Or load from settings if not passed
        if not api_key: raise ValueError("Cohere API key needed for CohereReranker.")
        return CohereReranker(api_key=api_key, **params)
    elif reranker_type == "cosine":
        if not embedding_client_for_reranker:
            raise ValueError("Embedding client (BaseLLM) must be provided for CosineReranker.")
        return CosineReranker(embedding_client=embedding_client_for_reranker, **params)
    elif reranker_type == "jina":
        api_key = api_key_for_reranker # Or load from settings
        if not api_key: raise ValueError("Jina API key needed for JinaReranker.")
        return JinaReranker(api_key=api_key, **params)
    elif reranker_type == "llm":
        if not llm_client_for_reranker:
            raise ValueError("LLM client (BaseLLM) must be provided for LLMReranker.")
        return LLMReranker(llm_client=llm_client_for_reranker, **params)
    else:
        raise ValueError(f"Unsupported reranker type: {reranker_type}")
