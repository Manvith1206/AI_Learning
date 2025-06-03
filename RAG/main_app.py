import streamlit as st
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any

# Attempt to import from rag_modular, with fallbacks for UI rendering if not found
try:
    from rag_modular.Common import RAG_Constants as constants
    from rag_modular.Common.config_manager import ConfigManager
    from rag_modular.Common.document import Document
    from rag_modular.Pipeline.rag_pipeline import RAGPipeline
    CHUNKERS = constants.ChunkerType
    EMBEDDERS = constants.EmbedderType
    LLMS = constants.GeminiLLMModel # Assuming Gemini, adjust if other LLMs are primary
    EVALUATORS = constants.EvaluatorType
    RETRIEVERS = constants.RetrieverType
    RERANKERS = constants.RerankerType
    LLM_SERVICES = constants.LLMServiceType
    RAG_MODULAR_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import from rag_modular: {e}. Some features will be disabled. Ensure rag_modular is installed and in PYTHONPATH.")
    # Mock objects for UI rendering
    class MockEnum:
        def __init__(self, *args):
            self._members = {arg: arg for arg in args}
        def __getattr__(self, name):
            return self._members.get(name, name) # Return name itself if not found
        def __iter__(self):
            return iter(self._members.keys())
        def list(self):
            return list(self._members.keys())

    CHUNKERS = MockEnum("RecursiveCharacterTextSplitter", "SemanticChunker")
    EMBEDDERS = MockEnum("GoogleGenerativeAIEmbeddings", "OpenAIEmbeddings")
    LLMS = MockEnum("gemini-pro", "gemini-1.5-pro-latest")
    EVALUATORS = MockEnum("RAGAS", "DeepEval")
    RETRIEVERS = MockEnum("similarity_search", "multi_query")
    RERANKERS = MockEnum("flashrank", "cohere")
    LLM_SERVICES = MockEnum("Google", "OpenAI")
    RAG_MODULAR_AVAILABLE = False
    ConfigManager = None
    RAGPipeline = None
    Document = None

from backend.pipeline_manager import PipelineManager
from session_manager import SessionManager
from ui.ui_components import UIComponents
from ui.metrics_display import MetricsDisplay

class ModularRAGApp:
    def __init__(self):
        self.session_manager = SessionManager()
        self.pipeline_manager = PipelineManager() # Initializes pipeline_config in session_state
        self.ui_components = UIComponents()
        self.metrics_display = MetricsDisplay()

    def _handle_config_change(self):
        st.session_state.pipeline_rebuild_needed = True
        st.session_state.pipeline_created = False
        st.session_state.pipeline_instance = None
        st.session_state.evaluation_results = {}
        st.session_state.last_query_metrics = {}
        st.info("Configuration changed. Pipeline will be rebuilt on next action.")

    def _update_pipeline_config(self, component_type: str, param_name: str, widget_key: str):
        value = st.session_state[widget_key] # Get value from session_state using the widget's key
        if RAG_MODULAR_AVAILABLE:
            try:
                st.session_state.pipeline_config.set_config_value(component_type, param_name, value)
                self._handle_config_change()
            except Exception as e:
                st.error(f"Error updating config for {component_type} ({param_name}={value}): {e}")
        else:
            st.warning("RAG_Modular library not available. Configuration changes are mocked.")
            if 'mock_pipeline_config' not in st.session_state:
                st.session_state.mock_pipeline_config = {}
            if component_type not in st.session_state.mock_pipeline_config:
                st.session_state.mock_pipeline_config[component_type] = {}
            st.session_state.mock_pipeline_config[component_type][param_name] = value
            self._handle_config_change()

    def _render_sidebar_config(self):
        with st.sidebar:
            st.header("⚙️ Pipeline Configuration")

            if not RAG_MODULAR_AVAILABLE:
                st.warning("RAG Modular library not loaded. Configuration options are illustrative.")
            
            # Helper to get current config value or default for widgets
            def get_current_value(component_key, param_key, default_value):
                if RAG_MODULAR_AVAILABLE and hasattr(st.session_state, 'pipeline_config') and st.session_state.pipeline_config:
                    return st.session_state.pipeline_config.get_config_value(component_key, param_key, default_value)
                elif 'mock_pipeline_config' in st.session_state and component_key in st.session_state.mock_pipeline_config:
                    return st.session_state.mock_pipeline_config[component_key].get(param_key, default_value)
                return default_value

            # LLM Configuration
            with st.expander("🤖 LLM Configuration", expanded=True):
                llm_options = LLM_SERVICES.list() if RAG_MODULAR_AVAILABLE else list(LLM_SERVICES)
                current_llm_service = get_current_value(constants.LLM_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'LLM', constants.CONFIG_SERVICE_NAME if RAG_MODULAR_AVAILABLE else 'service', llm_options[0])
                selected_llm_service = st.selectbox(
                    "LLM Service", options=llm_options,
                    index=llm_options.index(current_llm_service) if current_llm_service in llm_options else 0,
                    key="llm_service_selector",
                    on_change=self._update_pipeline_config, args=(constants.LLM_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'LLM', constants.CONFIG_SERVICE_NAME if RAG_MODULAR_AVAILABLE else 'service', "llm_service_selector")
                )

                llm_model_options = LLMS.list() if RAG_MODULAR_AVAILABLE else list(LLMS)
                current_llm_model = get_current_value(constants.LLM_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'LLM', constants.CONFIG_MODEL_NAME if RAG_MODULAR_AVAILABLE else 'model', llm_model_options[0])
                selected_llm = st.selectbox(
                    "LLM Model", options=llm_model_options,
                    index=llm_model_options.index(current_llm_model) if current_llm_model in llm_model_options else 0,
                    key="llm_selector",
                    on_change=self._update_pipeline_config, args=(constants.LLM_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'LLM', constants.CONFIG_MODEL_NAME if RAG_MODULAR_AVAILABLE else 'model', "llm_selector")
                )
                
                current_temp = get_current_value(constants.LLM_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'LLM', constants.CONFIG_TEMPERATURE if RAG_MODULAR_AVAILABLE else 'temperature', 0.7)
                temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=float(current_temp), step=0.01, key="llm_temp",
                                        on_change=self._update_pipeline_config, args=(constants.LLM_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'LLM', constants.CONFIG_TEMPERATURE if RAG_MODULAR_AVAILABLE else 'temperature', "llm_temp"))

            # Embedder Configuration
            with st.expander("↪️ Embedder Configuration", expanded=False):
                embedder_options = EMBEDDERS.list() if RAG_MODULAR_AVAILABLE else list(EMBEDDERS)
                current_embedder = get_current_value(constants.EMBEDDER_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Embedder', constants.CONFIG_TYPE_PARAM if RAG_MODULAR_AVAILABLE else 'type', embedder_options[0])
                selected_embedder = st.selectbox(
                    "Embedder Model", options=embedder_options,
                    index=embedder_options.index(current_embedder) if current_embedder in embedder_options else 0,
                    key="embedder_selector",
                    on_change=self._update_pipeline_config, args=(constants.EMBEDDER_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Embedder', constants.CONFIG_TYPE_PARAM if RAG_MODULAR_AVAILABLE else 'type', "embedder_selector")
                )

            # Chunker Configuration
            with st.expander("🔪 Chunker Configuration", expanded=False):
                chunker_options = CHUNKERS.list() if RAG_MODULAR_AVAILABLE else list(CHUNKERS)
                current_chunker = get_current_value(constants.CHUNKER_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Chunker', constants.CONFIG_TYPE_PARAM if RAG_MODULAR_AVAILABLE else 'type', chunker_options[0])
                selected_chunker = st.selectbox(
                    "Chunker Type", options=chunker_options,
                    index=chunker_options.index(current_chunker) if current_chunker in chunker_options else 0,
                    key="chunker_selector",
                    on_change=self._update_pipeline_config, args=(constants.CHUNKER_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Chunker', constants.CONFIG_TYPE_PARAM if RAG_MODULAR_AVAILABLE else 'type', "chunker_selector")
                )

                current_chunk_size = get_current_value(constants.CHUNKER_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Chunker', constants.CONFIG_CHUNK_SIZE if RAG_MODULAR_AVAILABLE else 'chunk_size', 1000)
                chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=int(current_chunk_size), step=50, key="chunk_size_selector",
                                             on_change=self._update_pipeline_config, args=(constants.CHUNKER_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Chunker', constants.CONFIG_CHUNK_SIZE if RAG_MODULAR_AVAILABLE else 'chunk_size', "chunk_size_selector"))

                current_chunk_overlap = get_current_value(constants.CHUNKER_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Chunker', constants.CONFIG_CHUNK_OVERLAP if RAG_MODULAR_AVAILABLE else 'chunk_overlap', 200)
                chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=int(current_chunk_overlap), step=50, key="chunk_overlap_selector",
                                                on_change=self._update_pipeline_config, args=(constants.CHUNKER_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Chunker', constants.CONFIG_CHUNK_OVERLAP if RAG_MODULAR_AVAILABLE else 'chunk_overlap', "chunk_overlap_selector"))

            # Retriever Configuration
            with st.expander("🔍 Retriever Configuration", expanded=False):
                retriever_options = RETRIEVERS.list() if RAG_MODULAR_AVAILABLE else list(RETRIEVERS)
                current_retriever = get_current_value(constants.RETRIEVER_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Retriever', constants.CONFIG_TYPE_PARAM if RAG_MODULAR_AVAILABLE else 'type', retriever_options[0])
                selected_retriever = st.selectbox(
                    "Retriever Type", options=retriever_options,
                    index=retriever_options.index(current_retriever) if current_retriever in retriever_options else 0,
                    key="retriever_selector",
                    on_change=self._update_pipeline_config, args=(constants.RETRIEVER_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Retriever', constants.CONFIG_TYPE_PARAM if RAG_MODULAR_AVAILABLE else 'type', "retriever_selector")
                )
                
                current_top_k = get_current_value(constants.RETRIEVER_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Retriever', constants.CONFIG_TOP_K if RAG_MODULAR_AVAILABLE else 'top_k', 5)
                top_k = st.number_input("Top K", min_value=1, max_value=20, value=int(current_top_k), step=1, key="retriever_top_k_selector",
                                        on_change=self._update_pipeline_config, args=(constants.RETRIEVER_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Retriever', constants.CONFIG_TOP_K if RAG_MODULAR_AVAILABLE else 'top_k', "retriever_top_k_selector"))

            # Reranker Configuration
            with st.expander("🔄 Reranker Configuration", expanded=False):
                reranker_options = RERANKERS.list() if RAG_MODULAR_AVAILABLE else list(RERANKERS)
                current_reranker = get_current_value(constants.RERANKER_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Reranker', constants.CONFIG_TYPE_PARAM if RAG_MODULAR_AVAILABLE else 'type', reranker_options[0])
                selected_reranker = st.selectbox(
                    "Reranker Type", options=reranker_options,
                    index=reranker_options.index(current_reranker) if current_reranker in reranker_options else 0,
                    key="reranker_selector",
                    on_change=self._update_pipeline_config, args=(constants.RERANKER_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Reranker', constants.CONFIG_TYPE_PARAM if RAG_MODULAR_AVAILABLE else 'type', "reranker_selector")
                )

                current_reranker_top_n = get_current_value(constants.RERANKER_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Reranker', constants.CONFIG_TOP_N if RAG_MODULAR_AVAILABLE else 'top_n', 3)
                reranker_top_n = st.number_input("Reranker Top N", min_value=1, max_value=10, value=int(current_reranker_top_n), step=1, key="reranker_top_n_selector",
                                               on_change=self._update_pipeline_config, args=(constants.RERANKER_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Reranker', constants.CONFIG_TOP_N if RAG_MODULAR_AVAILABLE else 'top_n', "reranker_top_n_selector"))

            # Evaluator Configuration
            with st.expander("⚖️ Evaluator Configuration", expanded=False):
                evaluator_options = EVALUATORS.list() if RAG_MODULAR_AVAILABLE else list(EVALUATORS)
                current_evaluator = get_current_value(constants.EVALUATOR_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Evaluator', constants.CONFIG_TYPE_PARAM if RAG_MODULAR_AVAILABLE else 'type', evaluator_options[0])
                selected_evaluator = st.selectbox(
                    "Evaluator Type", options=evaluator_options,
                    index=evaluator_options.index(current_evaluator) if current_evaluator in evaluator_options else 0,
                    key="evaluator_selector",
                    on_change=self._update_pipeline_config, args=(constants.EVALUATOR_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Evaluator', constants.CONFIG_TYPE_PARAM if RAG_MODULAR_AVAILABLE else 'type', "evaluator_selector")
                )

            st.markdown("---    ")
            if st.button("🔄 Run Full Configuration Test", key="run_full_test", help="Rebuilds pipeline and runs evaluation if evaluator is configured."):
                if RAG_MODULAR_AVAILABLE:
                    st.session_state.pipeline_rebuild_needed = True # Force rebuild with current settings
                    pipeline = self.pipeline_manager.get_or_create_pipeline()
                    if pipeline and get_current_value(constants.EVALUATOR_CONFIG_KEY if RAG_MODULAR_AVAILABLE else 'Evaluator', constants.CONFIG_TYPE_PARAM if RAG_MODULAR_AVAILABLE else 'type', None):
                        if st.session_state.documents:
                            with st.spinner("Running evaluation..."):
                                st.session_state.evaluation_results = pipeline.evaluate_retrieval(st.session_state.documents)
                            st.success("Full configuration test and evaluation complete!")
                        else:
                            st.warning("Please upload documents to run evaluation.")
                    elif pipeline:
                        st.success("Pipeline rebuilt with new configuration. No evaluator selected for full test.")
                    else:
                        st.error("Failed to build pipeline for the test.")
                else:
                    st.info("Mock full test run. RAG_Modular not available.")

    def _trigger_flashcard_generation(self):
        # Check if documents are processed or if a pipeline with a knowledge base exists
        pipeline = self.pipeline_manager.get_or_create_pipeline()
        can_generate_from_pipeline = RAG_MODULAR_AVAILABLE and pipeline and getattr(pipeline, 'vector_store', None) is not None
        can_generate_from_docs = st.session_state.get("processed_docs_count", 0) > 0

        if not can_generate_from_docs and not can_generate_from_pipeline:
            st.warning("Please upload and process documents first, or ensure the RAG pipeline has an existing knowledge base.")
            st.session_state.flashcards_generation_attempted = True # Mark attempt even if prerequisites fail
            st.session_state.flashcards = [] # Clear any old flashcards
            return

        self.pipeline_manager.generate_flashcards_from_docs()
        st.session_state.flashcards_generation_attempted = True

    def _handle_document_upload(self):
        uploaded_files = st.file_uploader("Upload your documents (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"], accept_multiple_files=True, key="doc_uploader")
        if uploaded_files:
            if st.button("Process Uploaded Documents", key="process_docs"):
                if RAG_MODULAR_AVAILABLE:
                    with st.spinner("Processing documents..."):
                        # This now correctly calls the PipelineManager method which updates session_state.documents
                        processed_docs_count, failed_files_count = self.pipeline_manager.process_uploaded_documents(uploaded_files)
                    if processed_docs_count > 0:
                        st.success(f"Successfully processed {processed_docs_count} documents.")
                    if failed_files_count > 0:
                        st.error(f"Failed to process {failed_files_count} files.")
                    if processed_docs_count == 0 and failed_files_count == 0:
                        st.info("No new documents were processed.")
                else:
                    st.info(f"Mock processing {len(uploaded_files)} documents. RAG_Modular not available.")
                    if 'documents' not in st.session_state: st.session_state.documents = []
                    for uploaded_file in uploaded_files:
                        # Simple mock document object
                        mock_doc = type('MockDoc', (), {'page_content': f"Mock content for {uploaded_file.name}", 'metadata': {'source': uploaded_file.name}})()
                        st.session_state.documents.append(mock_doc)
                    st.success(f"Mock processed {len(uploaded_files)} documents.")

    def run(self):
        self.ui_components.initialize_page()
        self.session_manager.initialize_session_state() # Ensures all keys are present
        self._render_sidebar_config() # Render sidebar and attach callbacks

        # Ensure pipeline is created/rebuilt if needed AFTER sidebar interactions might have changed config
        if RAG_MODULAR_AVAILABLE and (st.session_state.get("pipeline_rebuild_needed", True) or not st.session_state.get("pipeline_instance")):
            with st.spinner("Initializing RAG Pipeline..."):
                pipeline = self.pipeline_manager.get_or_create_pipeline()
                if pipeline:
                    st.session_state.pipeline_rebuild_needed = False
                    st.sidebar.success("Pipeline Ready!")
                else:
                    st.sidebar.error("Pipeline initialization failed!")
        elif not RAG_MODULAR_AVAILABLE:
            st.sidebar.info("RAG Pipeline (Mocked)")

        tab1, tab2, tab3 = self.ui_components.create_tabs()

        with tab1: # Chat with Documents
            st.header("💬 Chat with Your Documents")
            self._handle_document_upload()
            self.ui_components.render_chat_area() # Displays existing messages

            if prompt := st.chat_input("Ask a question about your documents..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                if not RAG_MODULAR_AVAILABLE:
                    with st.chat_message("assistant"):
                        st.markdown("Hello! RAG_Modular library is not available. I am a mock assistant.")
                    st.session_state.messages.append({"role": "assistant", "content": "Hello! RAG_Modular library is not available. I am a mock assistant."})
                elif not st.session_state.get("pipeline_instance"):
                    st.warning("Pipeline is not initialized. Please check configurations or process documents.")
                else:
                    with st.spinner("Thinking..."):
                        response_dict = self.pipeline_manager.process_query(prompt)
                        answer = response_dict.get("answer", "Sorry, I could not find an answer.")
                        retrieved_docs = response_dict.get("retrieved_documents", [])
                        step_metrics = response_dict.get("step_metrics", {})
                        st.session_state.last_query_metrics = step_metrics

                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        if retrieved_docs:
                            with st.expander("🔍 Retrieved Context"):
                                for i, doc_item in enumerate(retrieved_docs):
                                    # Assuming doc_item is a Document object from rag_modular or a compatible dict
                                    content = getattr(doc_item, 'page_content', doc_item.get('page_content', 'N/A'))
                                    metadata = getattr(doc_item, 'metadata', doc_item.get('metadata', {}))
                                    source = metadata.get('source', 'Unknown')
                                    st.markdown(f"**Document {i+1} (Source: {source})**")
                                    st.caption(content)
                                    st.markdown("---")
                    st.session_state.messages.append({"role": "assistant", "content": answer, "retrieved_docs": retrieved_docs})

        with metrics_tab:
            st.header("📊 Performance Metrics")
            if RAG_MODULAR_AVAILABLE:
                st.subheader("Overall Evaluation Metrics (from Full Test)")
                eval_results = st.session_state.get("evaluation_results")
                if eval_results and isinstance(eval_results, dict):
                    self.metrics_display.display_evaluation_metrics(eval_results)
                else:
                    st.info("No evaluation results available. Run a 'Full Configuration Test' from the sidebar with uploaded documents.")

                st.subheader("Last Query Performance")
                last_metrics = st.session_state.get("last_query_metrics")
                pipeline_inst = st.session_state.get("pipeline_instance")
                if last_metrics and isinstance(last_metrics, dict) and pipeline_inst:
                    for step_name, metrics_data in last_metrics.items():
                        if isinstance(metrics_data, dict): # Ensure metrics_data is a dict
                           self.metrics_display.display_pipeline_step_metrics(step_name, metrics_data, pipeline_inst)
                        else:
                            st.warning(f"Metrics for step '{step_name}' are not in the expected format.")
                elif not pipeline_inst:
                     st.info("Pipeline not initialized. Ask a question in the chat tab first.")
                else:
                    st.info("No metrics from the last query. Ask a question in the chat tab first.")
            else:
                st.info("RAG_Modular library not available. Metrics display is disabled.")

        with flashcards_tab:
            st.header("🗂️ FlashCards")
            st.write("Generate Q&A flashcards from your processed documents.")
            
            if st.button("Generate Flashcards from Processed Documents", key="generate_flashcards_btn", on_click=self._trigger_flashcard_generation):
                # Action is handled by the on_click callback, which calls _trigger_flashcard_generation
                # Feedback (success/error/info) will be displayed by generate_flashcards_from_docs in pipeline_manager
                pass # Button click itself doesn't need to do more here

            if 'flashcards' in st.session_state and st.session_state.flashcards:
                st.markdown("--- ")
                st.subheader("Generated Flashcards")
                for i, card in enumerate(st.session_state.flashcards):
                    with st.expander(f"Flashcard {i+1}: {card.get('question', 'N/A')}"):
                        st.markdown(f"**Answer:** {card.get('answer', 'N/A')}")
            elif st.session_state.get("flashcards_generation_attempted", False) and not st.session_state.get("flashcards"):
                 st.info("No flashcards were generated. Ensure documents are processed and the pipeline supports flashcard generation.")
            
            # Initial message if no documents processed and button not yet clicked
            if not st.session_state.get("processed_docs_count", 0) > 0 and not (RAG_MODULAR_AVAILABLE and self.pipeline_manager.get_or_create_pipeline() and getattr(self.pipeline_manager.get_or_create_pipeline(), 'vector_store', None)):
                 st.info("Please upload and process documents first to enable flashcard generation.")

    def run(self):
        self.ui_components.initialize_page()
        self.session_manager.initialize_session_state() # Ensures all keys are present
        self._render_sidebar_config() # Render sidebar and attach callbacks

        # Ensure pipeline is created/rebuilt if needed AFTER sidebar interactions might have changed config
        if RAG_MODULAR_AVAILABLE and (st.session_state.get("pipeline_rebuild_needed", True) or not st.session_state.get("pipeline_instance")):
            with st.spinner("Initializing RAG Pipeline..."):
                pipeline = self.pipeline_manager.get_or_create_pipeline()
                if pipeline:
                    st.session_state.pipeline_rebuild_needed = False
                    st.sidebar.success("Pipeline Ready!")
                else:
                    st.sidebar.error("Pipeline initialization failed!")
        elif not RAG_MODULAR_AVAILABLE:
            st.sidebar.info("RAG Pipeline (Mocked)")

        chat_tab, metrics_tab, flashcards_tab = self.ui_components.create_tabs()
        self._render_main_content(chat_tab, metrics_tab, flashcards_tab)

if __name__ == "__main__":
    app = ModularRAGApp()
    app.run()
