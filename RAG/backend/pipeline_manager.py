import streamlit as st
import logging

# Setup logger for this module
logger = logging.getLogger(__name__)

try:
    from rag_modular.Pipeline.rag_pipeline import RAGPipeline
    from rag_modular.Common.config_manager import ConfigManager
    RAG_MODULAR_AVAILABLE = True
except ImportError:
    RAG_MODULAR_AVAILABLE = False
    logger.warning("rag_modular library not found. Using mock implementations for RAGPipeline and ConfigManager.")

    class MockRAGPipeline:
        def __init__(self, config):
            logger.info("MockRAGPipeline initialized.")
            self.config = config

        def process_query(self, query: str):
            logger.info(f"MockRAGPipeline processing query: {query}")
            return "This is a mock response to your query.", ["Mock context 1", "Mock context 2"], {"retriever": {"time_taken": 0.1, "cost": 0}, "llm": {"time_taken": 0.5, "cost": 0.0001}}

        def process_uploaded_documents(self, files):
            logger.info(f"MockRAGPipeline processing {len(files)} uploaded documents.")
            st.session_state.processed_docs_count = len(files) # Simulate processing
            # In a real scenario, this might also populate some form of document cache or summary
            # that could be used by generate_flashcards if it needs explicit source_texts.
            if files:
                st.session_state.mock_uploaded_content_summary = [f"Content from {f.name[:20]}..." for f in files]
            return True

        def get_pipeline_step_metrics(self):
            logger.info("MockRAGPipeline get_pipeline_step_metrics called.")
            return {
                "retriever": {"time_taken": 0.12, "cost": 0.0},
                "reranker": {"time_taken": 0.05, "cost": 0.0},
                "llm": {"time_taken": 0.6, "cost": 0.00015}
            }
        
        def evaluate_retrieval(self, eval_data=None):
            logger.info("MockRAGPipeline evaluate_retrieval called.")
            return {"mock_metric_1": 0.9, "mock_metric_2": 0.75}

        def generate_flashcards(self, source_texts: list[str] = None):
            logger.info(f"MockRAGPipeline generating flashcards. Provided source texts: {len(source_texts) if source_texts else 'None'}")
            # If no source_texts, use generic mock flashcards
            # If source_texts are provided (e.g. from document summaries or chat), try to use them.
            base_flashcards = [
                {"question": "What is the capital of Mockland?", "answer": "Mockville"},
                {"question": "How does the mock pipeline work?", "answer": "Magically, with mock data!"}
            ]
            if source_texts:
                base_flashcards.append({
                    "question": f"What was in the first provided source text snippet?", 
                    "answer": f"Content similar to: {source_texts[0][:50]}..." if source_texts else "No specific source text snippet provided."
                })
            return base_flashcards

    class MockConfigManager:
        def __init__(self):
            logger.info("MockConfigManager initialized.")
            self._config = {}

        def get_config_value(self, component_type: str, param_name: str, default_value: any = None) -> any:
            return self._config.get(component_type, {}).get(param_name, default_value)

        def set_config_value(self, component_type: str, param_name: str, value: any):
            if component_type not in self._config:
                self._config[component_type] = {}
            self._config[component_type][param_name] = value
            logger.info(f"MockConfigManager: Set {component_type}.{param_name} = {value}")

    if not RAG_MODULAR_AVAILABLE:
        RAGPipeline = MockRAGPipeline
        ConfigManager = MockConfigManager

class PipelineManager:
    def __init__(self):
        if "pipeline_config" not in st.session_state or st.session_state.pipeline_config is None:
            logger.info("Initializing pipeline_config in session state.")
            st.session_state.pipeline_config = ConfigManager()
            st.session_state.pipeline_needs_rebuild = True
        
        if "pipeline_instance" not in st.session_state:
            st.session_state.pipeline_instance = None
        if "pipeline_created" not in st.session_state:
            st.session_state.pipeline_created = False
        if "pipeline_needs_rebuild" not in st.session_state:
            st.session_state.pipeline_needs_rebuild = True
        if "flashcards" not in st.session_state:
            st.session_state.flashcards = []

    def get_or_create_pipeline(self):
        if not st.session_state.get("pipeline_created", False) or \
           st.session_state.get("pipeline_instance") is None or \
           st.session_state.get("pipeline_needs_rebuild", False):
            
            if RAG_MODULAR_AVAILABLE:
                try:
                    st.info("Attempting to build RAG pipeline with current configuration...")
                    pipeline_config = st.session_state.pipeline_config
                    pipeline = RAGPipeline(pipeline_config)
                    st.session_state.pipeline_instance = pipeline
                    st.session_state.pipeline_created = True
                    st.session_state.pipeline_needs_rebuild = False
                    st.success("RAG pipeline built successfully!")
                    logger.info("RAG pipeline built successfully.")
                except Exception as e:
                    st.error(f"Failed to build RAG pipeline. Error: {e}")
                    logger.error(f"Failed to build RAG pipeline: {e}", exc_info=True)
                    st.session_state.pipeline_instance = None
                    st.session_state.pipeline_created = False
                    raise
            else:
                st.warning("RAG_Modular library not available. Using mock pipeline.")
                logger.warning("Using mock RAG pipeline as rag_modular library is not available.")
                st.session_state.pipeline_instance = RAGPipeline(st.session_state.pipeline_config)
                st.session_state.pipeline_created = True
                st.session_state.pipeline_needs_rebuild = False
        return st.session_state.get("pipeline_instance")

    def process_query(self, query: str):
        pipeline = self.get_or_create_pipeline()
        if not pipeline:
            st.error("Pipeline not available to process query.")
            logger.error("Pipeline not available in process_query")
            return "Error: Pipeline not available.", [], {}
        try:
            response, context, metrics = pipeline.process_query(query)
            st.session_state.last_query_metrics = metrics
            logger.info(f"Query processed. Response: {response[:50]}... Metrics: {metrics}")
            return response, context, metrics
        except Exception as e:
            st.error(f"Error during query processing: {e}")
            logger.error(f"Error during query processing: {e}", exc_info=True)
            return f"Error processing query: {e}", [], {}

    def get_pipeline_step_metrics(self):
        pipeline = st.session_state.get("pipeline_instance")
        if pipeline and hasattr(pipeline, 'get_pipeline_step_metrics'):
            try:
                metrics = pipeline.get_pipeline_step_metrics()
                logger.info(f"Retrieved pipeline step metrics: {metrics}")
                return metrics
            except Exception as e:
                st.error(f"Error retrieving pipeline step metrics: {e}")
                logger.error(f"Error retrieving pipeline step metrics: {e}", exc_info=True)
                return {}
        elif not RAG_MODULAR_AVAILABLE:
             logger.info("No pipeline instance, but RAG_MODULAR_AVAILABLE is False. Returning mock metrics.")
             return MockRAGPipeline(None).get_pipeline_step_metrics()
        logger.warning("Pipeline instance not available for get_pipeline_step_metrics")
        return {}

    def process_uploaded_documents(self, uploaded_files):
        pipeline = self.get_or_create_pipeline()
        if not pipeline:
            st.error("Pipeline not available to process documents.")
            logger.error("Pipeline not available in process_uploaded_documents")
            return False
        try:
            success = pipeline.process_uploaded_documents(uploaded_files)
            if success:
                st.success(f"{len(uploaded_files)} documents processed and added to the knowledge base.")
                logger.info(f"{len(uploaded_files)} documents processed successfully.")
            else:
                st.error("Failed to process uploaded documents.")
                logger.error("Failed to process uploaded documents according to pipeline.")
            return success
        except Exception as e:
            st.error(f"An error occurred while processing documents: {e}")
            logger.error(f"An error occurred while processing documents: {e}", exc_info=True)
            return False

    def generate_flashcards_from_docs(self):
        pipeline = self.get_or_create_pipeline()
        if not pipeline:
            st.error("Pipeline not available to generate flashcards.")
            logger.error("Pipeline not available in generate_flashcards_from_docs")
            st.session_state.flashcards = []
            return False

        flashcards = []
        try:
            if RAG_MODULAR_AVAILABLE:
                if hasattr(pipeline, 'generate_flashcards'):
                    st.info("Generating flashcards using the RAG pipeline...")
                    # The actual RAGPipeline.generate_flashcards might take arguments 
                    # like document_ids or use its internal state. For now, assume it takes no args.
                    flashcards = pipeline.generate_flashcards()
                else:
                    st.error("The configured RAG pipeline does not support flashcard generation (missing 'generate_flashcards' method).")
                    logger.error("RAGPipeline instance does not have 'generate_flashcards' method.")
                    st.session_state.flashcards = []
                    return False
            else: # Mock pipeline
                st.info("Generating mock flashcards...")
                # Provide some context to the mock if available from processed docs or chat
                source_texts_for_mock = st.session_state.get('mock_uploaded_content_summary', [])
                if not source_texts_for_mock and 'messages' in st.session_state:
                    source_texts_for_mock = [msg['content'] for msg in st.session_state.messages if msg['role'] == 'user'][-3:] # last 3 user messages
                if not source_texts_for_mock and st.session_state.get("processed_docs_count", 0) > 0:
                     source_texts_for_mock = ["Generic summary of processed document content for mock flashcards."]

                flashcards = pipeline.generate_flashcards(source_texts=source_texts_for_mock if source_texts_for_mock else None)

            st.session_state.flashcards = flashcards
            if flashcards:
                st.success(f"Generated {len(flashcards)} flashcards.")
                logger.info(f"Generated {len(flashcards)} flashcards: {flashcards}")
            else:
                st.info("No flashcards were generated by the pipeline.")
                logger.info("No flashcards generated by the pipeline.")
            return True
        except Exception as e:
            st.error(f"An error occurred during flashcard generation: {e}")
            logger.error(f"Error during flashcard generation: {e}", exc_info=True)
            st.session_state.flashcards = []
            return False
