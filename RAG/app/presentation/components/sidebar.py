import streamlit as st
from typing import Dict, List, Any, Callable, Tuple
from app.presentation.components.ui_components import UIComponents


class Sidebar:
    """Sidebar component for the RAG application"""
    
    def __init__(
        self,
        on_document_upload: Callable[[Any], Tuple[Any, List[Any]]],
        on_chunker_config_update: Callable[[Dict[str, Any]], None],
        on_embedder_config_update: Callable[[Dict[str, Any]], None],
        on_vector_store_config_update: Callable[[Dict[str, Any]], None],
        on_retriever_config_update: Callable[[Dict[str, Any]], None],
        on_reranker_config_update: Callable[[Dict[str, Any]], None],
        on_llm_config_update: Callable[[Dict[str, Any]], None],
        on_evaluator_config_update: Callable[[Dict[str, Any]], None],
        get_chunker_types: Callable[[], List[str]],
        get_embedder_types: Callable[[], List[str]],
        get_vector_store_types: Callable[[], List[str]],
        get_retriever_types: Callable[[], List[str]],
        get_reranker_types: Callable[[], List[str]],
        get_llm_types: Callable[[], List[str]],
        get_evaluator_types: Callable[[], List[str]],
        get_current_config: Callable[[str], Dict[str, Any]]
    ):
        """Initialize the sidebar component with callback functions"""
        self.on_document_upload = on_document_upload
        self.on_chunker_config_update = on_chunker_config_update
        self.on_embedder_config_update = on_embedder_config_update
        self.on_vector_store_config_update = on_vector_store_config_update
        self.on_retriever_config_update = on_retriever_config_update
        self.on_reranker_config_update = on_reranker_config_update
        self.on_llm_config_update = on_llm_config_update
        self.on_evaluator_config_update = on_evaluator_config_update
        self.get_chunker_types = get_chunker_types
        self.get_embedder_types = get_embedder_types
        self.get_vector_store_types = get_vector_store_types
        self.get_retriever_types = get_retriever_types
        self.get_reranker_types = get_reranker_types
        self.get_llm_types = get_llm_types
        self.get_evaluator_types = get_evaluator_types
        self.get_current_config = get_current_config
    
    def render(self):
        """Render the sidebar"""
        with st.sidebar:
            st.subheader("Configuration")
            self._render_config_tabs()
            self._render_upload_file_section()
    
    def _render_config_tabs(self):
        """Render configuration tabs in sidebar"""
        config_tabs = st.tabs([
            "Text Processing", 
            "Retrieval", 
            "Evaluation"
        ])
        
        with config_tabs[0]:
            self._render_text_processing_config()
        with config_tabs[1]:
            self._render_retrieval_config()
        with config_tabs[2]:
            self._render_evaluation_config()
    
    def _render_text_processing_config(self):
        """Render text processing configuration options"""
        st.write("**Text Processing**")
        
        # Chunker configuration
        chunker_types = self.get_chunker_types()
        current_chunker_config = self.get_current_config("chunker")
        chunker_type = st.selectbox(
            "Chunker Type",
            options=chunker_types,
            index=chunker_types.index(current_chunker_config["type"])
        )
        
        chunker_params = {}
        if chunker_type == "recursive":
            chunk_size = st.slider("Chunk Size", 10, 10000, current_chunker_config["params"].get("chunk_size", 150))
            chunk_overlap = st.slider("Chunk Overlap", 0, 3000, current_chunker_config["params"].get("chunk_overlap", 70))
            chunker_params = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
        elif chunker_type == "semantic":
            min_chunk_size = st.number_input("Min Chunk Size", 0, 10000, current_chunker_config["params"].get("min_chunk_size", 600))
            max_chunk_size = st.number_input("Max Chunk Size", 0, 10000, current_chunker_config["params"].get("max_chunk_size", 110))
            similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, current_chunker_config["params"].get("similarity_threshold", 0.65), 0.01)
            model_name = st.selectbox(
                "Model Name",
                options=["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2"],
                index=0
            )
            chunker_params = {
                "min_chunk_size": min_chunk_size,
                "max_chunk_size": max_chunk_size,
                "similarity_threshold": similarity_threshold,
                "model_name": model_name
            }
        elif chunker_type == "sentence":
            max_sentences = st.slider("Max Sentences", 1, 20, current_chunker_config["params"].get("max_sentences", 5))
            chunker_params = {"max_sentences": max_sentences}
        
        UIComponents.display_divider()
        
        # Vector store configuration
        vector_store_types = self.get_vector_store_types()
        current_vector_store_config = self.get_current_config("vector_store")
        vector_store_type = st.selectbox(
            "Vector Store",
            options=vector_store_types,
            index=vector_store_types.index(current_vector_store_config["type"])
        )
        
        vector_store_params = {}
        if vector_store_type == "scikit_learn":
            vector_store_params = {"metric": "cosine"}
        
        UIComponents.display_divider()
        
        # Embedder configuration
        embedder_types = self.get_embedder_types()
        current_embedder_config = self.get_current_config("embedder")
        embedder_type = st.selectbox(
            "Embedder Type",
            options=embedder_types,
            index=embedder_types.index(current_embedder_config["type"])
        )
        
        embedder_params = {}
        if embedder_type != "tfidf":
            # Simplified model selection - in a real app, this would be dynamic based on embedder type
            model_options = ["models/embedding-001", "text-embedding-ada-002", "embed-english-v3.0"]
            default_model = current_embedder_config["params"].get("model", model_options[0])
            model_index = model_options.index(default_model) if default_model in model_options else 0
            
            model = st.selectbox(
                "Embedding Model",
                options=model_options,
                index=model_index
            )
            embedder_params = {"model": model}
        
        # Apply button
        if st.button("Apply Text Processing Params", key="apply_text_proc"):
            with UIComponents.display_spinner("Applying Text Processing Params"):
                chunker_config = {"type": chunker_type, "params": chunker_params}
                vector_store_config = {"type": vector_store_type, "params": vector_store_params}
                embedder_config = {"type": embedder_type, "params": embedder_params}
                
                self.on_chunker_config_update(chunker_config)
                self.on_vector_store_config_update(vector_store_config)
                self.on_embedder_config_update(embedder_config)
                
                UIComponents.display_success("Text processing configuration updated.")
    
    def _render_retrieval_config(self):
        """Render retrieval configuration options"""
        st.write("**Retrieval**")
        
        # Retriever configuration
        retriever_types = self.get_retriever_types()
        current_retriever_config = self.get_current_config("retriever")
        retriever_type = st.selectbox(
            "Retriever Type",
            options=retriever_types,
            index=retriever_types.index(current_retriever_config["type"])
        )
        
        top_k = st.slider("Top-K-Docs for Retrieval", 1, 20, current_retriever_config["params"].get("top_k", 5))
        retriever_params = {"top_k": top_k}
        
        if retriever_type == "similarity":
            similarity_threshold = st.slider(
                "Similarity Threshold", 
                0.0, 1.0, 
                current_retriever_config["params"].get("similarity_threshold", 0.0), 
                0.01
            )
            retriever_params["similarity_threshold"] = similarity_threshold
        elif retriever_type == "hybrid":
            keyword_weight = st.slider(
                "Keyword Weight", 
                0.0, 1.0, 
                current_retriever_config["params"].get("keyword_weight", 0.3), 
                0.05
            )
            retriever_params["keyword_weight"] = keyword_weight
        
        UIComponents.display_divider()
        
        # Reranker configuration
        reranker_types = self.get_reranker_types()
        current_reranker_config = self.get_current_config("reranker")
        reranker_type = st.selectbox(
            "Re-ranker Type",
            options=reranker_types,
            index=reranker_types.index(current_reranker_config["type"])
        )
        
        top_k_rerank = st.slider(
            "Top-K-Docs for Re-ranking", 
            1, 20, 
            current_reranker_config["params"].get("top_k", 5), 
            key="top_k_rerank"
        )
        
        reranker_params = {"top_k": top_k_rerank}
        
        if reranker_type in ["llm", "cohere", "jina"]:
            # Simplified model selection - in a real app, this would be dynamic based on reranker type
            model_options = ["gemini-pro", "claude-3-haiku", "rerank-english-v2.0"]
            default_model = current_reranker_config["params"].get("model", model_options[0])
            model_index = model_options.index(default_model) if default_model in model_options else 0
            
            model = st.selectbox(
                "Reranker Model",
                options=model_options,
                index=model_index
            )
            reranker_params["model"] = model
        
        # Apply button
        if st.button("Apply Retrieval and Reranker Params", key="apply_retrieval"):
            with UIComponents.display_spinner("Applying Retrieval and Reranker Params"):
                retriever_config = {"type": retriever_type, "params": retriever_params}
                reranker_config = {"type": reranker_type, "params": reranker_params}
                
                self.on_retriever_config_update(retriever_config)
                self.on_reranker_config_update(reranker_config)
                
                UIComponents.display_success("Retrieval and reranking configuration updated.")
    
    def _render_evaluation_config(self):
        """Render evaluation configuration options"""
        st.write("**Evaluation**")
        
        # Evaluator configuration
        evaluator_types = self.get_evaluator_types()
        current_evaluator_config = self.get_current_config("evaluator")
        evaluator_type = st.selectbox(
            "Evaluator Type",
            options=evaluator_types,
            index=evaluator_types.index(current_evaluator_config["type"])
        )
        
        # Apply button
        if st.button("Apply Evaluation Params", key="apply_evaluation"):
            with UIComponents.display_spinner("Applying Evaluation Params"):
                evaluator_config = {"type": evaluator_type, "params": {}}
                
                self.on_evaluator_config_update(evaluator_config)
                
                UIComponents.display_success("Evaluation configuration updated.")
    
    def _render_upload_file_section(self):
        """Render file upload section in sidebar"""
        st.subheader("Upload and Process Documents")
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["pdf", "csv", "txt", "docx"],
            accept_multiple_files=False
        )
        
        if uploaded_file:
            if st.button("Process Document"):
                with UIComponents.display_spinner("Processing document..."):
                    documents, chunks = self.on_document_upload(uploaded_file)
                    
                    if documents and chunks:
                        st.session_state.documents = documents
                        st.session_state.chunks = chunks
                        UIComponents.display_success(f"Processed {len(chunks)} chunks from document")
                    else:
                        UIComponents.display_warning("No valid content was extracted from the document")
