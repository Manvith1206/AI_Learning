import streamlit as st
import os
import sys
import pandas as pd

def main():
    st.set_page_config(
        page_title="RAG Modular",
        page_icon=":notebook:",
        layout="wide"
    )

main()
# Create a loading indicator for initial setup
with st.spinner("Loading application..."):
    # Lazy import dependencies only when needed
    from rag_modular.RAG_Constants import (
        ChunkerType, EmbedderType,
        RetrieverType, RerankerType,
        EvaluatorType, GeminiLLMModel
    )
    import rag_modular.RAG_Constants as constants
    
    # Add rag_modular to path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_modular'))
    
    # Only import these when the app is fully loaded
    @st.cache_resource
    def load_pipeline_dependencies():
        from rag_modular.rag_pipeline import RAGPipeline
        from rag_modular.config_manager import ConfigManager
        import rag_modular.recursive_chunker
        import rag_modular.test_rag_pipeline
        import test_rag_combinations
        return {
            "RAGPipeline": RAGPipeline,
            "ConfigManager": ConfigManager,
            "test_rag_combinations": test_rag_combinations
        }
    
    # Load dependencies with caching
    deps = load_pipeline_dependencies()
    RAGPipeline = deps["RAGPipeline"]
    ConfigManager = deps["ConfigManager"]
    test_rag_combinations = deps["test_rag_combinations"]

TEMP_DIR = "temp_docs"

# Initialize session state for pipeline - but only create the actual pipeline when needed
if "pipeline" not in st.session_state:
    config_manager = ConfigManager()
    # Defer actual pipeline creation until needed
    st.session_state.pipeline_config = config_manager
    st.session_state.pipeline_created = False

if "documents" not in st.session_state:
    st.session_state.documents = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "LLM_Model_Options" not in st.session_state:
    st.session_state.LLM_Model_Options = [e.value for e in GeminiLLMModel]

# Function to lazily initialize the pipeline when needed
def get_pipeline():
    if not st.session_state.get("pipeline_created", False):
        with st.spinner("Initializing RAG pipeline..."):
            st.session_state.pipeline = RAGPipeline(st.session_state.pipeline_config)
            st.session_state.pipeline_created = True
    return st.session_state.pipeline

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
        st.divider()
        # Vector Store
        vector_store = st.selectbox(
            constants.VECTOR_STORE_DISPLAY_NAME,
            options=[e.value for e in constants.VectorStore],
            index=0            
        )
        if vector_store == constants.VectorStore.SCIKIT_LEARN.value:
            vector_store_params = {constants.CONFIG_TYPE_PARAM: constants.CONFIG_VECTOR_STORE_SKLEARN, constants.CONFIG_VECTOR_STORE_METRIC: {constants.CONFIG_METRIC_COSINE}}
        elif vector_store == constants.VectorStore.PINE_CONE.value:
            vector_store_params = {constants.CONFIG_TYPE_PARAM: constants.CONFIG_VECTOR_STORE_PINCONE}
        else:
            vector_store_params = {constants.CONFIG_TYPE_PARAM: constants.CONFIG_VECTOR_STORE_FAISS}
        st.divider()
        # Embedder selection
        embedder_type = st.selectbox(
            constants.EMBEDDER_TYPE_DISPLAY_NAME,
            options=[e.value for e in EmbedderType],
            index=0
        )
        
        if embedder_type != EmbedderType.TFIDF.value:
            emb_options = []
            if embedder_type == EmbedderType.COHERE.value:
                emb_options = constants.CohereEmbedModels
            elif embedder_type == EmbedderType.VOYAGE.value:
                emb_options = constants.VoyageEmbedModels
            elif embedder_type == EmbedderType.GEMINI.value:
                emb_options = constants.GeminiEmbedModels
            elif embedder_type == EmbedderType.MISTRAL.value:
                emb_options = constants.MISTRAL_EMBED_MODELS
            emb_model = st.selectbox(constants.EMBED_MODEL_DISPLAY_NAME, options=[e.value for e in emb_options])
            embedder_params = {constants.CONFIG_TYPE_PARAM: embedder_type, constants.CONFIG_MODEL: emb_model}
        else:
            embedder_params={constants.CONFIG_TYPE_PARAM: embedder_type}
        
        with st.spinner("Applying Text Processing Params"):
            # Apply text processing config
            if st.button("Apply Text Processing Params", key="apply_text_proc"):
                chunker_config = {constants.CONFIG_TYPE_PARAM: chunker_type, constants.CONFIG_PARAM: chunker_params}
                get_pipeline().update_component(constants.CONFIG_CHUNKER, chunker_config)            
                get_pipeline().update_component(constants.CONFIG_EMBEDDER, embedder_params)
                
                get_pipeline().update_component(constants.CONFIG_VECTOR_STORE, vector_store_params)
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
        elif re_ranker_type == RerankerType.JINA.value:
            re_ranker_params = {constants.CONFIG_TYPE_PARAM: RerankerType.JINA.value}
            st.session_state.LLM_Model_Options = [e.value for e in constants.JINA_RERANKER_MODELS]
        if re_ranker_params[constants.CONFIG_TYPE_PARAM] == RerankerType.LLM.value or re_ranker_params[constants.CONFIG_TYPE_PARAM] == RerankerType.COHERE.value or re_ranker_params[constants.CONFIG_TYPE_PARAM] == RerankerType.JINA.value:
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
        # elif retriever_type == RetrieverType.SENTENCE_WINDOW.value:  # sentence window
        #     window_size = st.number_input(constants.WINDOW_SIZE_DISPLAY_NAME, min_value=0, max_value=10)
        #     retriever_params = {constants.CONFIG_WINDOW_SIZE: window_size}
            
        
        # Top-k setting
        top_k = st.slider("Top-K Documents", 1, 20, 5)
        # Apply retrieval config
        with st.spinner("Applying Retrieval Params"):
            if st.button("Apply Retrieval Params", key="apply_retrieval"):
                retriever_config = {constants.CONFIG_TYPE_PARAM: retriever_type, constants.CONFIG_PARAM: retriever_params, constants.CONFIG_TOP_K_PARAM: top_k}
                if re_ranker_type == RerankerType.LLM.value:
                    service = st.selectbox("LLM Service", options=[e.value for e in constants.LLMServiceType])
                    reranker_config = {constants.CONFIG_TYPE_PARAM: re_ranker_type, constants.CONFIG_PARAM: model}
                elif re_ranker_type == RerankerType.COHERE.value or re_ranker_type == RerankerType.JINA.value:
                    reranker_config = {constants.CONFIG_TYPE_PARAM: re_ranker_type, constants.CONFIG_PARAM: model}
                else:
                    reranker_config = {constants.CONFIG_TYPE_PARAM: re_ranker_type, constants.CONFIG_PARAM: {}}

                get_pipeline().update_component(constants.CONFIG_RETRIEVER, retriever_config)
                get_pipeline().update_component(constants.CONFIG_RERANKER, reranker_config)
                st.success("Retrieval configuration updated.")
    
    with config_tabs[2]:
        st.write("**" + constants.EVALUATION_DISPLAY_NAME + "**")
        # Evaluator selection
        evaluator_type = st.selectbox(
            "Evaluator Type",
            options=[e.value for e in EvaluatorType],
            index=1
        )
        with st.spinner("Applying Evaluation Params"):
            # Apply evaluation config
            if st.button("Apply Evaluation Params", key="apply_evaluation"):
                evaluator_config = {constants.CONFIG_TYPE_PARAM: evaluator_type}
                get_pipeline().update_component(constants.CONFIG_EVALUATOR, evaluator_config)
                st.success("Evaluation configuration updated.")

    st.divider()
    st.write("**" + constants.CHAT_RESPONSE_CONFIG_DISPLAY_NAME + "**")
    # Chat response config
    llm_service = st.selectbox(constants.LLM_CHAT_SERVICE, options=[e.value for e in constants.LLMServiceType], index=0)
    llm_model_options = []
    if (llm_service == constants.LLMServiceType.GEMINI.value):
        llm_model_options = {model.display_name: model for model in constants.GeminiLLMModel}
    elif (llm_service == constants.LLMServiceType.CLAUDE.value):
        llm_model_options = {model.display_name: model for model in constants.CLAUDE_MODELS}
    else:
        llm_model_options = {model.display_name: model for model in constants.GeminiLLMModel}

    user_selected_llm_model = st.selectbox(constants.LLM_CHAT_SERVICE, options=llm_model_options.keys(), index=0)
    chat_response_config = {constants.CONFIG_TYPE_PARAM: llm_service, constants.CONFIG_MODEL: llm_model_options[user_selected_llm_model].value}
    
    if st.button("Apply Chat Response Config", key="apply_chat_response"):
        get_pipeline().update_component(constants.CONFIG_LLM, chat_response_config)
        st.success("Chat response configuration updated.")

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
                # Only initialize pipeline when needed
                pipeline = get_pipeline()
                texts = pipeline.extractText(uploaded_file)
                documents, chunks = pipeline.process_document(uploaded_file, texts)
                    
                if documents and chunks:
                    st.session_state.documents = documents
                    st.session_state.chunks = chunks
                    st.success(f"Processed {len(documents)} chunks from document")
                else:
                    st.warning("No valid content was extracted from the document")
    
    # Evaluation section
    st.subheader("Evaluation")
    ground_truth = st.text_area(constants.GROUND_TRUTH_DISPLAY_NAME, value=constants.GROUND_TRUTH_DEFAULT_VALUE)
    if st.button("Evaluate Last Query"):
        # Initialize pipeline when needed
        pipeline = get_pipeline()
        if hasattr(pipeline, constants.LAST_QUERY):
            try:
                metrics = pipeline.evaluate(ground_truths=ground_truth)
                
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
                
                overallScore = overallScore / metrics_df.count()
                st.write("Overall Score: " + str(overallScore))
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
    with st.expander("Config Details"):
        if get_pipeline().config_manager:
            Config_Content = f"Chunker Config: {get_pipeline().config_manager.config[constants.CONFIG_CHUNKER]}\nEmbedder Config: {get_pipeline().config_manager.config[constants.CONFIG_EMBEDDER]}\nVector Store Config{get_pipeline().config_manager.config[constants.CONFIG_VECTOR_STORE]}\nRetreiver Config: {get_pipeline().config_manager.config[constants.CONFIG_RETRIEVER]}\nLLM Config: {get_pipeline().config_manager.config[constants.CONFIG_LLM]}\nRe Ranking Config: {get_pipeline().config_manager.config[constants.CONFIG_RERANKER]}\n{get_pipeline().config_manager.config[constants.CONFIG_EVALUATOR]}"
            st.markdown(Config_Content)

    if st.button("Test All Configurations", key="test_all_combinations"):
        # Only import and run when button is clicked
        test_rag_combinations.run_tests()

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
                # Initialize pipeline when needed
                pipeline = get_pipeline()
                response = pipeline.query(prompt)
                st.markdown(f"**Re-ranking Explanation:**\n{response['rerank_explanation']}")
                st.markdown(response["answer"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"]
                })
        else:
            st.error("Please upload and process documents first.")
