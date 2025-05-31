import streamlit as st
import os
import sys
import pandas as pd
from typing import Dict, List, Any, Optional
from rag_modular.Common.rag_pipeline import RAGPipeline
from rag_modular.Common.config_manager import ConfigManager
import rag_modular.Common.RAG_Constants as constants
from rag_modular.Common.RAG_Constants import (
    ChunkerType, EmbedderType,GeminiLLMModel, EvaluatorType, RetrieverType, RerankerType)

class UIComponents:
    @staticmethod
    def initialize_page():
        st.set_page_config(
            page_title="RAG Modular",
            page_icon=":notebook:",
            layout="wide"
        )
    
    @staticmethod
    def create_tabs():
        return st.tabs(["Chat with Documents", "Performance Metrics", "FlashCards"])

class DocumentProcessor:
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        
    def process_uploaded_file(self, uploaded_file) -> tuple:
        """Process an uploaded document and return documents and chunks"""
        texts = self.pipeline.extractText(uploaded_file)
        return self.pipeline.process_document(uploaded_file, texts)

class MetricsDisplay:
    @staticmethod
    def display_metrics(metrics: Dict[str, float]):
        """Display evaluation metrics in a formatted way"""
        st.write("**Evaluation Metrics:**")
        metrics_df = pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Score": list(metrics.values())
        })
        st.dataframe(metrics_df)
        overall_score = sum(metrics.values()) / len(metrics)
        st.write(f"Overall Score: {overall_score}")
        st.bar_chart(metrics_df.set_index("Metric"))
        return metrics_df

class RAGApp:
    def __init__(self):
        self.ui = UIComponents()
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize all session state variables"""
        if "pipeline" not in st.session_state:
            config_manager = ConfigManager()
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
        if "LLM_Service" not in st.session_state:
            st.session_state.LLM_Service = constants.LLMServiceType.GEMINI.value

    def run(self):
        """Main application entry point"""
        self.ui.initialize_page()
        tab1, tab2 = self.ui.create_tabs()
        
        with tab1:
            self.render_chat_interface()
        with tab2:
            self.render_metrics_interface()

    def get_pipeline(self):
        """Get or initialize the pipeline"""
        if not st.session_state.get("pipeline_created", False):
            with st.spinner("Initializing RAG pipeline..."):
                st.session_state.pipeline = RAGPipeline(st.session_state.pipeline_config)
                st.session_state.pipeline_created = True
        return st.session_state.pipeline

    def render_chat_interface(self):
        """Render the chat interface tab"""
        with st.spinner("Loading application..."):
            self.load_dependencies()
        
        self.render_sidebar()
        self.render_chat_area()

    def render_metrics_interface(self):
        """Render the metrics interface tab"""
        RagPipelineStepToGetCostAndTimeDict = self.get_pipeline_metrics()
        
        st.write("This section provides the performance and cost metrics for each step in the RAG pipeline.")
        st.write("The metrics include time taken for processing and estimated cost for each step.")
        st.write("Note: The cost is an estimate based on the current configuration and may vary based on actual usage.")
        
        for key in RagPipelineStepToGetCostAndTimeDict.keys():
            self.display_pipeline_step_metrics(key, RagPipelineStepToGetCostAndTimeDict[key])

    def load_dependencies(self):
        """Load required dependencies"""
        from rag_modular.Common.RAG_Constants import (
            ChunkerType, EmbedderType,
            RetrieverType, RerankerType,
            EvaluatorType, GeminiLLMModel
        )
        import rag_modular.Common.RAG_Constants as constants
        sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_modular'))

    def get_pipeline_metrics(self):
        """Get metrics for all pipeline steps"""
        pipeline = self.get_pipeline()
        return {
            constants.CONFIG_CHUNKER: pipeline.get_chunker_cost_and_time(),
            constants.CONFIG_EMBEDDER: pipeline.get_embedder_cost_and_time(),
            constants.CONFIG_RETRIEVER: pipeline.get_retriever_cost_and_time(),
            constants.CONFIG_RERANKER: pipeline.get_reranker_cost_and_time(),
            constants.CONFIG_EVALUATOR: pipeline.get_evaluator_cost_and_time(),
            constants.CONFIG_VECTOR_STORE: pipeline.get_vector_store_cost_and_time(),
            constants.CONFIG_LLM: pipeline.get_llm_service_cost_and_time()
        }

    def display_pipeline_step_metrics(self, key: str, metrics: tuple):
        """Display metrics for a single pipeline step"""
        cost, time_taken = metrics
        with st.expander(f"📊 {key} - Performance & Cost", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("🕒 Time Taken", time_taken)
            col2.metric("💲 Estimated Cost", cost)
            cfg = self.get_pipeline().config_manager.get_config(key)
            name = cfg.get(constants.CONFIG_TYPE_PARAM)
            col3.markdown(f"RAG Pipeline Step Name: \n{name}")

    def render_chat_area(self):
        """Render the main chat area"""
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
                self.process_chat_input(prompt)

    def process_chat_input(self, prompt: str):
        """Process user chat input and generate response"""
        if st.session_state.documents:
            with st.spinner("🤔 Thinking...", show_time=True):
                pipeline = self.get_pipeline()
                response = pipeline.query(prompt)
                st.markdown(f"**Re-ranking Explanation:**\n{response['rerank_explanation']}")
                # st.markdown(response["answer"])
                # st.session_state.messages.append({
                #     "role": "assistant",
                #     "content": response["answer"]
                # })
        else:
            st.error("Please upload and process documents first.")

    def render_sidebar(self):
        """Render the sidebar with all configuration options"""
        with st.sidebar:
            st.subheader("Configuration")
            self.render_config_tabs()
            self.render_chat_response_config()
            self.render_upload_file_section()
            self.render_test_all_configs_section()

    def render_config_tabs(self):
        """Render configuration tabs in sidebar"""
        config_tabs = st.tabs([
            constants.TEXT_PROCESSING_DISPLAY_NAME, 
            constants.RETRIEVAL_DISPLAY_NAME, 
            constants.EVALUATION_DISPLAY_NAME
        ])
        
        with config_tabs[0]:
            self.render_text_processing_config()
        with config_tabs[1]:
            self.render_retrieval_config()
        with config_tabs[2]:
            self.render_evaluation_config()
            self.render_evaluation_section()

    def render_text_processing_config(self):
        """Render text processing configuration options"""
        st.write(f"**{constants.TEXT_PROCESSING_DISPLAY_NAME}**")
        options, index = self.get_ui_options(option_type=ChunkerType, config_name=constants.CONFIG_CHUNKER)
        chunker_type = st.selectbox(
            constants.CHUNKER_TYPE_DISPLAY_NAME,
            options=options,
            index=index
        )
        
        
        chunker_params = self.get_chunker_config(chunker_type)
        vector_store = self.get_vector_store_config()
        embedder_params = self.get_embedder_config()
        
        with st.spinner("Applying Text Processing Params"):
            if st.button("Apply Text Processing Params", key="apply_text_proc"):
                self.apply_text_processing_config(chunker_params, vector_store, embedder_params)

    def render_retrieval_config(self):
        """Render retrieval configuration options"""
        st.write(f"**{constants.RETRIEVAL_DISPLAY_NAME}**")
        
        retriever_config = self.get_retriever_config()
        reranker_config = self.get_reranker_config()

        with st.spinner("Applying Retrieval and Reranker Params"):
            if st.button("Applying Retrieval and Reranker Params", key="apply_retrieval"):
                self.apply_Retrieval_and_Reranker_config(retriever_config, reranker_config)

    def render_evaluation_config(self):
        """Render evaluation configuration options"""
        st.write(f"**{constants.EVALUATION_DISPLAY_NAME}**")
        
        options, index = self.get_ui_options(option_type=constants.EvaluatorType, config_name=constants.CONFIG_EVALUATOR)

        evaluator_type = st.selectbox(
            "Evaluator Type",
            options=options,
            index=index
        )
        
        evaluator_config = {constants.CONFIG_TYPE_PARAM: evaluator_type}
        
        with st.spinner("Applying Evaluation Params"):
            if st.button("Apply Evaluation Params", key="apply_evaluation"):
                self.get_pipeline().update_component(constants.CONFIG_EVALUATOR, evaluator_config)
                st.success("Evaluation configuration updated.")

    def get_ui_options(self, option_type, config_name: str):
        options = [e.value for e in option_type]
        config = st.session_state.pipeline_config.get_config(config_name)
        index = options.index(config[constants.CONFIG_TYPE_PARAM])
        return options,index

    def render_evaluation_section(self):
        # Evaluation section
        st.subheader("Evaluation")
        ground_truth = st.text_area(constants.GROUND_TRUTH_DISPLAY_NAME, value=constants.GROUND_TRUTH_DEFAULT_VALUE)
        if st.button("Evaluate Last Query"):
            # Initialize pipeline when needed
            pipeline = self.get_pipeline()
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

    def render_upload_file_section(self):
        """Render file upload section in sidebar"""
        st.subheader("Upload and Process Documents")
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["pdf", "csv", "txt", "docx"],
            accept_multiple_files=False
        )
        doc_processor = DocumentProcessor(self.get_pipeline())
        if uploaded_file:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    # Only initialize pipeline when needed
                    pipeline = self.get_pipeline()
                    documents, chunks = doc_processor.process_uploaded_file(uploaded_file)
                    
                    if documents and chunks:
                        st.session_state.documents = documents
                        st.session_state.chunks = chunks
                        st.success(f"Processed {len(documents)} chunks from document")
                    else:
                        st.warning("No valid content was extracted from the document")

    def get_retriever_config(self) -> dict[str, Any]:

        options, index = self.get_ui_options(option_type=constants.RetrieverType, config_name=constants.CONFIG_RETRIEVER)

        retriever_type = st.selectbox(
            "Retriever Type",
            options=options,
            index=index
        )

        top_k = st.slider("Top-K-Docs for Retrieval", 1, 20, 5)
        retriever_params = {}
        if retriever_type == RetrieverType.SIMILARITY.value:
            similarity_threshold = st.slider(constants.SIMILARITY_THRESHOLD_DISPLAY_NAME, 0.0, 1.0, 0.0, 0.01)
            retriever_params = {constants.CONFIG_SIMILARITY_THRESHOLD_PARAM: similarity_threshold, constants.CONFIG_TOP_K_PARAM: top_k}
        elif retriever_type == RetrieverType.HYBRID.value:
            keyword_weight = st.slider(constants.KEYWORD_WEIGHT_DISPLAY_NAME, 0.0, 1.0, 0.3, 0.05)
            retriever_params = {constants.CONFIG_KEYWORD_WEIGHT: keyword_weight, constants.CONFIG_TOP_K_PARAM: top_k}
        
        retriever_config = {
            constants.CONFIG_TYPE_PARAM: retriever_type,
            constants.CONFIG_PARAM: retriever_params
        }

        return retriever_config
    
    def get_reranker_config(self) -> dict[str, Any]:
        options, index = self.get_ui_options(option_type=constants.RerankerType, config_name=constants.CONFIG_RERANKER)

        re_ranker_type = st.selectbox(
        "Re-ranker Type",
        options=options,
        index=index
        )

        top_k_for_reranking = st.slider("Top-K-Docs for Re-ranking", 1, 20, 5, key="top_k_rerank")
        re_ranker_params = {
            constants.CONFIG_TOP_K_FOR_RERANKING_PARAM: top_k_for_reranking
        }
        
        if re_ranker_type in [RerankerType.LLM.value, RerankerType.COHERE.value, RerankerType.JINA.value]:
            model = st.selectbox(
                constants.MODEL_NAME_DISPLAY_NAME,
                options=self.get_reranker_model_options(
                reranker_type=re_ranker_type),
                index=0)
            
            re_ranker_params[constants.CONFIG_MODEL] = model
        
        reranker_config = {
            constants.CONFIG_TYPE_PARAM: re_ranker_type,
            constants.CONFIG_PARAM: re_ranker_params
        }

        return reranker_config

    def render_chat_response_config(self):
        """Render chat response configuration"""
        st.divider()
        st.write(f"**{constants.CHAT_RESPONSE_CONFIG_DISPLAY_NAME}**")
        options, index = self.get_ui_options(option_type=constants.LLMServiceType, config_name=constants.CONFIG_LLM)

        llm_service = st.selectbox(
            constants.LLM_CHAT_SERVICE, 
            options=options, 
            index=index
        )
        st.session_state.LLM_Service = llm_service
        
        llm_model_options = self.get_llm_model_options(llm_service)

        options = list(llm_model_options.keys())
        default_option = st.session_state.pipeline_config.get_config(constants.CONFIG_LLM)[constants.CONFIG_PARAM][constants.CONFIG_MODEL]

        # Get the index
        index = options.index(default_option)
        user_selected_llm_model = st.selectbox(
            constants.LLM_CHAT_SERVICE, 
            options=llm_model_options.keys(), 
            index=index
        )
        
        chat_response_config = {
            constants.CONFIG_TYPE_PARAM: llm_service,
            constants.CONFIG_PARAM: {
                constants.CONFIG_MODEL: llm_model_options[user_selected_llm_model]
            }
        }
        
        if st.button("Apply Chat Response Config", key="apply_chat_response"):
            self.get_pipeline().update_component(constants.CONFIG_LLM, chat_response_config)
            st.success("Chat response configuration updated.")

    def render_test_all_configs_section(self):
        st.divider()
        st.subheader("Test All Configurations")
        st.write("Click the button below to test all configurations with different combinations of chunkers, embedder, vector store, and reranker.")
        import rag_modular.Testing.rag_evaluation_v2 as test_rag_combinations
        if st.button("Test All Configurations", key="test_all_combinations"):
            # Only import and run when button is clicked
            test_rag_combinations.test_rag_combinations()

    def get_chunker_config(self, chunker_type: str) -> dict:
        """Get parameters for the selected chunker type"""
        chunker_params = {}
        if chunker_type == ChunkerType.RECURSIVE.value:
            chunk_size = st.slider(constants.CHUNK_SIZE_DISPLAY_NAME, 10, 10000, 150)
            chunk_overlap = st.slider(constants.CHUNK_OVERLAP_DISPLAY_NAME, 0, 3000, 70)
            chunker_params = {
                constants.CONFIG_CHUNK_SIZE_PARAM: chunk_size,
                constants.CONFIG_CHUNK_OVERLAP_PARAM: chunk_overlap
            }
        elif chunker_type == ChunkerType.SEMANTIC.value:
            min_chunk_size = st.number_input(constants.MIN_CHUNK_SIZE_DISPLAY_NAME, 0, 10000, 600)
            max_chunk_size = st.number_input(constants.MAX_CHUNK_SIZE_DISPLAY_NAME, 0, 10000, 110)
            similarity_threshold = st.text_area(constants.SIMILARITY_THRESHOLD_DISPLAY_NAME, 0.65)
            model_name = st.selectbox(
                constants.MODEL_NAME_DISPLAY_NAME,
                options=[
                    constants.SENTENCE_TRANSFORMER_MODEL_ALL_MINI,
                    constants.SENTENCE_TRANSFORMER_MODEL_PARAPHRASE_MINI
                ]
            )
            chunker_params = {
                constants.CONFIG_MIN_CHUNK_SIZE_DISPLAY_NAME: min_chunk_size,
                constants.CONFIG_MAX_CHUNK_SIZE_DISPLAY_NAME: max_chunk_size,
                constants.CONFIG_SIMILARITY_THRESHOLD_PARAM: float(similarity_threshold),
                constants.CONFIG_MODEL_NAME: model_name
            }
        elif chunker_type == ChunkerType.SENTENCE.value:
            max_sentences = st.slider(constants.MAX_SENTENCES_DISPLAY_NAME, 1, 20, 5)
            chunker_params = {constants.CONFIG_MAX_SENTENCES: max_sentences}
        chunker_config = {
            constants.CONFIG_TYPE_PARAM: chunker_type,
            constants.CONFIG_PARAM: chunker_params
        }
        return chunker_config

    def get_vector_store_config(self):
        """Configure vector store settings"""
        st.divider()

        options, index = self.get_ui_options(option_type=constants.VectorStore, config_name=constants.CONFIG_VECTOR_STORE)

        vector_store = st.selectbox(
            constants.VECTOR_STORE_DISPLAY_NAME,
            options=options,
            index=index
        )
        
        if vector_store == constants.VectorStore.SCIKIT_LEARN.value:
            return {
                constants.CONFIG_TYPE_PARAM: constants.VectorStore.SCIKIT_LEARN.value,
                constants.CONFIG_PARAM: {
                    constants.CONFIG_VECTOR_STORE_METRIC: constants.CONFIG_METRIC_COSINE
                }
            }
        elif vector_store == constants.VectorStore.PINE_CONE.value:
            return {constants.CONFIG_TYPE_PARAM: constants.VectorStore.PINE_CONE.value}
        elif vector_store == constants.VectorStore.CHROMA.value:
            return {constants.CONFIG_TYPE_PARAM: constants.VectorStore.CHROMA.value}
        else:
            return {constants.CONFIG_TYPE_PARAM: constants.VectorStore.FAISS.value}

    def get_embedder_config(self):
        """Configure embedder settings"""
        st.divider()
        options, index = self.get_ui_options(option_type=constants.EmbedderType, config_name=constants.CONFIG_EMBEDDER)

        embedder_type = st.selectbox(
            constants.EMBEDDER_TYPE_DISPLAY_NAME,
            options=options,
            index=index
        )
        
        if embedder_type != EmbedderType.TFIDF.value:
            emb_options = self.get_embedder_options(embedder_type)
            emb_model = st.selectbox(
                constants.EMBED_MODEL_DISPLAY_NAME,
                options=[e.value for e in emb_options]
            )
            return {
                constants.CONFIG_TYPE_PARAM: embedder_type,
                constants.CONFIG_PARAM: {constants.CONFIG_MODEL: emb_model}
            }
        return {constants.CONFIG_TYPE_PARAM: embedder_type}

    def get_embedder_options(self, embedder_type: str):
        """Get embedder model options based on embedder type"""
        if embedder_type == EmbedderType.COHERE.value:
            return constants.CohereEmbedModels
        elif embedder_type == EmbedderType.VOYAGE.value:
            return constants.VoyageEmbedModels
        elif embedder_type == EmbedderType.GEMINI.value:
            return constants.GeminiEmbedModels
        elif embedder_type == EmbedderType.MISTRAL.value:
            return constants.MISTRAL_EMBED_MODELS
        return []
    
    def get_llm_model_options(self, llm_service: str) -> dict:
        """Get LLM model options based on the selected service"""
        if llm_service == constants.LLMServiceType.GEMINI.value:
            return {model.display_name: model.value for model in constants.GeminiLLMModel}
        elif llm_service == constants.LLMServiceType.CLAUDE.value:
            return {model.display_name: model.value for model in constants.CLAUDE_MODELS}
        else:
            return {model.display_name: model.value for model in constants.GeminiLLMModel}
    
    def get_reranker_model_options(self, reranker_type: str) -> dict:
        """Get re-ranker model options based on the selected type"""
        if reranker_type == RerankerType.LLM.value:
            return self.get_llm_model_options(st.session_state.LLM_Service)
        elif reranker_type == RerankerType.COHERE.value:
            return {model.value: model for model in constants.CohereLLMModel}
        elif reranker_type == RerankerType.JINA.value:
            return {model.value: model for model in constants.JINA_RERANKER_MODELS}
        return {}
    
    def apply_text_processing_config(self, chunker_params, vector_store, embedder_params):
        """Apply text processing configuration to the pipeline"""
        self.get_pipeline().update_component(constants.CONFIG_CHUNKER, chunker_params)
        self.get_pipeline().update_component(constants.CONFIG_EMBEDDER, embedder_params)
        self.get_pipeline().update_component(constants.CONFIG_VECTOR_STORE, vector_store)
        
        st.success("Text processing configuration updated.")
    
    def apply_Retrieval_and_Reranker_config(self, retriever_config, re_ranker_config):
        """Apply text processing configuration to the pipeline"""
        self.get_pipeline().update_component(constants.CONFIG_RETRIEVER, retriever_config)
        self.get_pipeline().update_component(constants.CONFIG_RERANKER, re_ranker_config)
        
        st.success("Retrieval and Reranking configuration updated.")

def main():
    app = RAGApp()
    app.run()

if __name__ == "__main__":
    main()
