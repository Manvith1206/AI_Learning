import streamlit as st
import os
import sys
import pandas as pd
from rag_modular.RAG_Constants import (
    ChunkerType, EmbedderType,
    RetrieverType, RerankerType,
    EvaluatorType, GeminiLLMModel
)
import rag_modular.RAG_Constants as constants
# Add rag_modular to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_modular'))
from rag_modular.rag_pipeline import RAGPipeline
from rag_modular.config_manager import ConfigManager

TEMP_DIR = "temp_docs"

# Initialize session state for pipeline
if "pipeline" not in st.session_state:
    config_manager = ConfigManager()
    st.session_state.pipeline = RAGPipeline(config_manager)

if "documents" not in st.session_state:
    st.session_state.documents = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "LLM_Model_Options" not in st.session_state:
    st.session_state.LLM_Model_Options = [e.value for e in constants.GeminiLLMModel]

import streamlit as st

# Sidebar for configuration and document upload
with st.sidebar:
    st.subheader("Configuration")
    
    # Create tabs for different component types
    config_tabs = st.tabs([constants.TEXT_PROCESSING_DISPLAY_NAME, constants.RETRIEVAL_DISPLAY_NAME, constants.EVALUATION_DISPLAY_NAME])
    
    with config_tabs[0]:
        st.write("**" + constants.TEXT_PROCESSING_DISPLAY_NAME + "**")
        # Chunker selection
        chunker_type = st.selectbox(
            constants.CHUNKER_TYPE_DISPLAY_NAME,
            options=[e.value for e in ChunkerType],
            index=0
        )
        
        # Chunker parameters
        if chunker_type == ChunkerType.RECURSIVE.value:
            chunk_size = st.slider(constants.CHUNK_SIZE_DISPLAY_NAME, 100, 10000, 600)
            chunk_overlap = st.slider(constants.CHUNK_OVERLAP_DISPLAY_NAME, 0, 3000, 200)
            chunker_params = {constants.CONFIG_CHUNK_SIZE_PARAM: chunk_size, constants.CONFIG_CHUNK_OVERLAP_PARAM: chunk_overlap}
        elif chunker_type == ChunkerType.SEMANTIC.value:
            min_chunk_size = st.slider(constants.MIN_CHUNK_SIZE_DISPLAY_NAME, 100, 10000, 600)
            max_chunk_size = st.slider(constants.MAX_CHUNK_SIZE_DISPLAY_NAME, 100, 10000, 600)
            similarity_threshold = st.text_area(constants.SIMILARITY_THRESHOLD_DISPLAY_NAME, 0.65)
            model_name = st.selectbox(constants.MODEL_NAME_DISPLAY_NAME, 
                                      options=[constants.SENTENCE_TRANSFORMER_MODEL_ALL_MINI, 
                                      constants.SENTENCE_TRANSFORMER_MODEL_PARAPHRASE_MINI])
            chunker_params = {constants.CONFIG_MIN_CHUNK_SIZE_DISPLAY_NAME: min_chunk_size, 
                              constants.CONFIG_MAX_CHUNK_SIZE_DISPLAY_NAME: max_chunk_size, 
                              constants.CONFIG_SIMILARITY_THRESHOLD_PARAM: float(similarity_threshold), 
                              constants.CONFIG_MODEL_NAME: model_name}
        elif chunker_type == ChunkerType.SENTENCE.value:  
            max_sentences = st.slider(constants.MAX_SENTENCES_DISPLAY_NAME, 1, 20, 5)
            chunker_params = {constants.CONFIG_MAX_SENTENCES: max_sentences}
            
        # Embedder selection
        embedder_type = st.selectbox(
            constants.EMBEDDER_TYPE_DISPLAY_NAME,
            options=[e.value for e in EmbedderType],
            index=0
        )

        if embedder_type == EmbedderType.COHERE.value or embedder_type == EmbedderType.VOYAGE.value:
            if embedder_type == EmbedderType.COHERE.value:
                emb_model = st.selectbox(constants.EMBED_MODEL_DISPLAY_NAME, options=[constants.COHERE_EMBED_MODEL_DEFAULT, constants.COHERE_EMBED_MODEL_ENG])
            elif embedder_type == EmbedderType.VOYAGE.value:
                emb_model = st.selectbox(constants.EMBED_MODEL_DISPLAY_NAME, options=[e.value for e in constants.VoyageEmbedModels])
            embedder_params = {constants.CONFIG_TYPE_PARAM: embedder_type, constants.CONFIG_MODEL: emb_model}
        else:
            embedder_params={constants.CONFIG_TYPE_PARAM: embedder_type}
        
        # Apply text processing config
        if st.button("Apply Text Processing", key="apply_text_proc"):
            chunker_config = {constants.CONFIG_TYPE_PARAM: chunker_type, constants.CONFIG_PARAM: chunker_params}
            st.session_state.pipeline.update_component(constants.CONFIG_CHUNKER, chunker_config)
            st.session_state.pipeline.update_component(constants.CONFIG_VECTOR_STORE, vector_store_params)
            
            st.session_state.pipeline.update_component(constants.CONFIG_CHUNKER, chunker_config)            
            st.session_state.pipeline.update_component(constants.CONFIG_EMBEDDER, embedder_params)
            st.success("Text processing configuration updated.")
    
    with config_tabs[1]:
        st.write("**" + constants.RETRIEVAL_DISPLAY_NAME + "**")
        # Retriever selection
        retriever_type = st.selectbox(
            "Retriever Type",
            options=[e.value for e in RetrieverType],
            index=0
        )
        re_ranker_type = st.selectbox(
            "Re-ranker Type",
            options=[e.value for e in RerankerType],
            index=0
        )

        
        if re_ranker_type == RerankerType.COSINE.value: # cosine
            re_ranker_params = {constants.CONFIG_TYPE_PARAM: RerankerType.COSINE.value}
        elif re_ranker_type == RerankerType.LLM.value:  # llm
            re_ranker_params = {constants.CONFIG_TYPE_PARAM: RerankerType.LLM.value}
            st.session_state.LLM_Model_Options = [e.value for e in constants.GeminiLLMModel]
        elif re_ranker_type == RerankerType.COHERE.value:# cohere
            re_ranker_params = {constants.CONFIG_TYPE_PARAM: RerankerType.COHERE.value}
            st.session_state.LLM_Model_Options = [e.value for e in constants.CohereLLMModel]

        if re_ranker_params[constants.CONFIG_TYPE_PARAM] == RerankerType.LLM.value or re_ranker_params[constants.CONFIG_TYPE_PARAM] == RerankerType.COHERE.value:
            model = st.selectbox(constants.MODEL_NAME_DISPLAY_NAME, options=[e for e in st.session_state.LLM_Model_Options], index=0)
            re_ranker_params[constants.CONFIG_MODEL] = model

        # Retriever parameters
        if retriever_type == RetrieverType.SIMILARITY.value:
            similarity_threshold = st.slider(constants.SIMILARITY_THRESHOLD_DISPLAY_NAME, 0.0, 1.0, 0.0, 0.01)
            # Only pass similarity_threshold to retriever
            retriever_params = {constants.CONFIG_SIMILARITY_THRESHOLD_PARAM: similarity_threshold}
        elif retriever_type == RetrieverType.HYBRID.value:  # hybrid
            keyword_weight = st.slider(constants.KEYWORD_WEIGHT_DISPLAY_NAME, 0.0, 1.0, 0.3, 0.05)
            retriever_params = {constants.CONFIG_KEYWORD_WEIGHT: keyword_weight}
        elif retriever_type == RetrieverType.SENTENCE_WINDOW.value:  # sentence window
            window_size = st.number_input(constants.WINDOW_SIZE_DISPLAY_NAME, min_value=0, max_value=10)
            retriever_params = {constants.CONFIG_WINDOW_SIZE: window_size}
            
        
        # Top-k setting
        top_k = st.slider("Top-K Documents", 1, 20, 5)
        # Apply retrieval config
        if st.button("Apply Retrieval", key="apply_retrieval"):
            retriever_config = {constants.CONFIG_TYPE_PARAM: retriever_type, constants.CONFIG_PARAM: retriever_params, constants.CONFIG_TOP_K_PARAM: top_k}
            if re_ranker_type == RerankerType.LLM.value:
                service = st.selectbox("LLM Service", options=[e.value for e in constants.LLMServiceType])

                reranker_config = {constants.CONFIG_TYPE_PARAM: re_ranker_type, constants.CONFIG_PARAM: model}
            elif re_ranker_type == RerankerType.COHERE.value:

                reranker_config = {constants.CONFIG_TYPE_PARAM: re_ranker_type, constants.CONFIG_PARAM: model}
            else:
                reranker_config = {constants.CONFIG_TYPE_PARAM: re_ranker_type, constants.CONFIG_PARAM: {}}

            st.session_state.pipeline.update_component(constants.CONFIG_RETRIEVER, retriever_config)
            st.session_state.pipeline.update_component(constants.CONFIG_RERANKER, reranker_config)
            st.success("Retrieval configuration updated.")
    
    with config_tabs[2]:
        st.write("**" + constants.EVALUATION_DISPLAY_NAME + "**")
        # Evaluator selection
        evaluator_type = st.selectbox(
            "Evaluator Type",
            options=[e.value for e in EvaluatorType],
            index=0
        )
        # Apply evaluation config
        if st.button("Apply Evaluation", key="apply_evaluation"):
            evaluator_config = {constants.CONFIG_TYPE_PARAM: evaluator_type}
            st.session_state.pipeline.update_component(constants.CONFIG_EVALUATOR, evaluator_config)
            st.success("Evaluation configuration updated.")

    # Document upload
    st.subheader("Upload and Process Documents")
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "csv", "txt", "docx"],
        accept_multiple_files=False
    )
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    documents, chunks = st.session_state.pipeline.process_document(uploaded_file)
                    
                    if documents and chunks:
                        st.session_state.documents = documents
                        st.session_state.chunks = chunks
                        st.success(f"Processed {len(documents)} chunks from document")
                    else:
                        st.warning("No valid content was extracted from the document")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
    
    # Evaluation section
    st.subheader("Evaluation")
    ground_truth = st.text_area(constants.GROUND_TRUTH_DISPLAY_NAME, value=constants.GROUND_TRUTH_DEFAULT_VALUE)
    if st.button("Evaluate Last Query"):
        if hasattr(st.session_state.pipeline, constants.LAST_QUERY):
            try:
                metrics = st.session_state.pipeline.evaluate(ground_truths=ground_truth)
                
                # Display metrics in a nice format
                st.write("**Evaluation Metrics:**")
                
                metrics_df = pd.DataFrame({
                    "Metric": list(metrics.keys()),
                    "Score": list(metrics.values())
                })
                st.dataframe(metrics_df)
                overallScore = 0
                for score in list(metrics.values()):
                    overallScore += score
                st.write("Overall Score: " + overallScore)
                # Show a bar chart of metrics
                st.bar_chart(metrics_df.set_index("Metric"))
                
                # Store metrics in session state
                st.session_state.last_evaluation = metrics
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")
        else:
            st.warning("No query to evaluate. Ask a question first.")
    
   
    # Show previous evaluation if available
    if "last_evaluation" in st.session_state:
        with st.expander("Previous Evaluation Results"):
            metrics_df = pd.DataFrame({
                "Metric": list(st.session_state.last_evaluation.keys()),
                "Score": list(st.session_state.last_evaluation.values())
            })
            st.dataframe(metrics_df)

    st.divider()
    if st.session_state.pipeline.config_manager:
        Config_Content = f"Chunker Config: {st.session_state.pipeline.config_manager.config[constants.CONFIG_CHUNKER]}\nEmbedder Config: {st.session_state.pipeline.config_manager.config[constants.CONFIG_EMBEDDER]}\nVector Store Config{st.session_state.pipeline.config_manager.config[constants.CONFIG_VECTOR_STORE]}\nRetreiver Config: {st.session_state.pipeline.config_manager.config[constants.CONFIG_RETRIEVER]}\nLLM Config: {st.session_state.pipeline.config_manager.config[constants.CONFIG_LLM]}\nRe Ranking Config: {st.session_state.pipeline.config_manager.config[constants.CONFIG_RERANKER]}\n{st.session_state.pipeline.config_manager.config[constants.CONFIG_EVALUATOR]}"
        st.markdown(Config_Content)

# Main chat interface
st.subheader("Chat with your Documents")
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if st.session_state.documents:
            with st.spinner("Thinking..."):
                response = st.session_state.pipeline.query(prompt)
                st.markdown(f"**Re-ranking Explanation:**\n{response['rerank_explanation']}")
                st.markdown(response["answer"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"]
                })
        else:
            st.error("Please upload and process documents first.")

