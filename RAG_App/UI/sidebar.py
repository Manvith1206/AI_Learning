from typing import Dict, List, Any, Callable, Tuple
from UI.UI_Components import UIComponents
from Utils import Exceptions
from infrastructure.Common.rag_pipeline import RAGPipeline
from infrastructure.Common.RAG_Constants import ChunkerType, EmbedderType, RetrieverType, RerankerType
from infrastructure.Common import RAG_Constants as constants
import pandas as pd

from services.services import DocumentProcessor
from UI.flashcard_display import FlashcardDisplay
from config import ConfigManager
import Utils.Utils
from Utils.cache_manager import CacheManager

class Sidebar:
    """Sidebar component for the RAG application"""
    def __init__(self):
        """Initialize the sidebar with all required use cases"""
        UIComponents.initialize_session_state(
            {
            "messages": [],
            "documents": None,
            "chunks": None,
            "LLM_Model_Options": [e.value for e in constants.GeminiLLMModel],
            "LLM_Service": constants.LLMServiceType.GEMINI.value,
        }
        )
    
    @staticmethod
    def initialize_page():
        UIComponents.initialize_page()
    
    @staticmethod
    def create_tabs():
        return UIComponents.create_tabs(["Chat with Documents", "Performance Metrics", "FlashCards"])
    
    def run(self):
        """Main application entry point"""
        self.initialize_page()
        tab1, tab2, tab3 = self.create_tabs()
        
        with tab1:
            self.render_chat_interface()
        with tab2:
            self.render_metrics_interface()
        with tab3:
            self.render_flashcards()
    
    def render_flashcards(self):
        """Render the flashcard display interface"""
        flashcardDisplay = FlashcardDisplay(on_generate_flashcards=[])
        flashcardDisplay.render()

    def render_chat_interface(self):
        """Render the chat interface tab"""
        with UIComponents.display_spinner("Loading application..."):
            self.render_sidebar()
            self.render_chat_area()

    def render_metrics_interface(self):
        """Render the metrics interface tab"""
        RagPipelineStepToGetCostAndTimeDict = self.get_pipeline_metrics()
        
        UIComponents.write("This section provides the performance and cost metrics for each step in the RAG pipeline.")
        UIComponents.write("The metrics include time taken for processing and estimated cost for each step.")
        UIComponents.write("Note: The cost is an estimate based on the current configuration and may vary based on actual usage.")
        
        for key in RagPipelineStepToGetCostAndTimeDict.keys():
            self.display_pipeline_step_metrics(key, RagPipelineStepToGetCostAndTimeDict[key])

    def get_pipeline_metrics(self):
        """Get metrics for all pipeline steps"""
        pipeline = Utils.Utils.get_pipeline()
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
        with UIComponents.create_expander(f"📊 {key} - Performance & Cost", expanded=False):
            col1, col2, col3 = UIComponents.create_columns(3)
            col1.metric("🕒 Time Taken", time_taken)
            col2.metric("💲 Estimated Cost", cost)
            cfg = Utils.Utils.get_pipeline().config_manager.get_config(key)
            name = cfg.get(constants.CONFIG_TYPE_PARAM)
            col3.markdown(f"RAG Pipeline Step Name: \n{name}")

    def render_chat_area(self):
        """Render the main chat area"""
        UIComponents.create_subheader_UI("Chat with your Documents")
        
        # Display chat history
        for message in UIComponents.get_session_state_messages():
            UIComponents.display_message_with_role(role=message["role"], message=message['content'])

        # Chat input
        if prompt := UIComponents.chat_input("Ask a question about your documents", key="chat_input"):
            UIComponents.add_message_to_chat(role='user', content=prompt)
            UIComponents.display_message_with_role(role='user', message=prompt)
            UIComponents.process_chat_input(role='assistant', content=prompt, pipeline=Utils.Utils.get_pipeline(), prompt=prompt)
            
    def render_sidebar(self):
        """Render the sidebar with all configuration options"""
        with UIComponents.create_sidebar():
            UIComponents.create_subheader_UI("Configuration")
            self.render_config_tabs()
            self.render_chat_response_config()
            self.render_upload_file_section()
            self.render_test_all_configs_section()

    def render_config_tabs(self):
        """Render configuration tabs in sidebar"""
        config_tabs = UIComponents.create_tabs([
            constants.TEXT_PROCESSING_DISPLAY_NAME, 
            constants.RETRIEVAL_DISPLAY_NAME
        ])
        
        with config_tabs[0]:
            self.render_text_processing_config()
        with config_tabs[1]:
            self.render_retrieval_config()
        # with config_tabs[2]:
        #     self.render_evaluation_config()
        #     self.render_evaluation_section()

    def render_text_processing_config(self):
        """Render text processing configuration options"""
        UIComponents.write(f"**{constants.TEXT_PROCESSING_DISPLAY_NAME}**")
        options, index = self.get_ui_options(option_type=ChunkerType, config_name=constants.CONFIG_CHUNKER)
        chunker_type = UIComponents.selectbox(
            constants.CHUNKER_TYPE_DISPLAY_NAME,
            options=options,
            index=index
        )

        chunker_params = self.get_chunker_config(chunker_type)
        vector_store = self.get_vector_store_config()
        embedder_params = self.get_embedder_config()

        with UIComponents.display_spinner("Applying Text Processing Config"):
            if UIComponents.create_button("Apply Text Processing Config"):
                self.apply_text_processing_config(chunker_params, vector_store, embedder_params)

    def render_retrieval_config(self):
        """Render retrieval configuration options"""
        UIComponents.write(f"**{constants.RETRIEVAL_DISPLAY_NAME}**")
        
        retriever_config = self.get_retriever_config()
        reranker_config = self.get_reranker_config()

        with UIComponents.display_spinner("Applying Retrieval and Reranker Params"):
            if UIComponents.create_button("Applying Retrieval and Reranker Params", key="apply_retrieval"):
                self.apply_Retrieval_and_Reranker_config(retriever_config, reranker_config)

    def render_evaluation_config(self):
        """Render evaluation configuration options"""
        UIComponents.write(f"**{constants.EVALUATION_DISPLAY_NAME}**")
        
        options, index = self.get_ui_options(option_type=constants.EvaluatorType, config_name=constants.CONFIG_EVALUATOR)

        evaluator_type = UIComponents.selectbox(
            "Evaluator Type",
            options=options,
            index=index
        )
        
        evaluator_config = {constants.CONFIG_TYPE_PARAM: evaluator_type}
        
        with UIComponents.display_spinner("Applying Evaluation Params"):
            if UIComponents.create_button("Apply Evaluation Params", key="apply_evaluation"):
                Utils.Utils.get_pipeline().update_component(constants.CONFIG_EVALUATOR, evaluator_config)
                UIComponents.display_success("Evaluation configuration updated.")

    def get_ui_options(self, option_type, config_name: str):
        options = [e.value for e in option_type]
        st_config = UIComponents.get_session_state_variable("pipeline_config")
        config = UIComponents.get_session_state_variable("pipeline_config").get_config(config_name)
        index = options.index(config[constants.CONFIG_TYPE_PARAM])
        return options, index

    def render_evaluation_section(self):
        # Evaluation section
        UIComponents.create_subheader_UI("Evaluation")
        ground_truth = UIComponents.create_text_area(constants.GROUND_TRUTH_DISPLAY_NAME, value=constants.GROUND_TRUTH_DEFAULT_VALUE, key="ground_truth_input")
        if UIComponents.create_button("Evaluate Last Query", key="evaluate_last_query"):
            # Initialize pipeline when needed
            pipeline = Utils.Utils.get_pipeline()
            if hasattr(pipeline, constants.LAST_QUERY):
                try:
                    metrics = pipeline.evaluate(ground_truths=ground_truth)
                    
                    # Display metrics in a nice format
                    UIComponents.write("**Evaluation Metrics:**")
                    
                    metrics_df = pd.DataFrame({
                        "Metric": list(metrics.keys()),
                        "Score": list(metrics.values())
                    })
                    UIComponents.display_dataframe(metrics_df)
                    overallScore = 0
                    for score in list(metrics.values()):
                        overallScore += score
                    
                    overallScore = overallScore / metrics_df.count()
                    UIComponents.write("Overall Score: " + str(overallScore))
                    # Show a bar chart of metrics
                    UIComponents.display_bar_chart(metrics_df.set_index("Metric"))

                    # Store metrics in session state
                    UIComponents.set_session_state_variable("last_evaluation", metrics)
                except Exception as e:
                    UIComponents.display_error(f"Error during evaluation: {str(e)}")
            else:
                UIComponents.display_warning("No query to evaluate. Ask a question first.")

    def render_upload_file_section(self):
        """Render file upload section in sidebar with caching."""
        UIComponents.create_subheader_UI("Upload Documents")
        uploaded_file = UIComponents.create_file_uploader(
            "Upload a document to start",
            file_types=["pdf", "docx", "txt", "csv"]
        )

        if uploaded_file:
            pipeline = Utils.Utils.get_pipeline()
            cache_manager = CacheManager()

            # Get current configuration for caching
            chunker_config = pipeline.config_manager.get_config(constants.CONFIG_CHUNKER)
            embedder_config = pipeline.config_manager.get_config(constants.CONFIG_EMBEDDER)
            vector_store_config = pipeline.config_manager.get_config(constants.CONFIG_VECTOR_STORE)
            
            processing_params = {
                "chunker": chunker_config,
                "embedder": embedder_config,
                "vector_store": vector_store_config
            }

            file_bytes = uploaded_file.getvalue()
            cache_key = cache_manager.generate_cache_key(file_bytes, processing_params)
            
            cached_data = cache_manager.load_from_cache(cache_key)
            
            if cached_data:
                with UIComponents.display_spinner("Loading processed document from cache..."):
                    # Restore both vector_store and the fitted embedder
                    pipeline.vector_store = cached_data['vector_store']
                    pipeline.embedder = cached_data['embedder']
                    
                    UIComponents.set_session_state_variable("vector_store", pipeline.vector_store)
                    documents = pipeline.vector_store.documents
                    UIComponents.set_session_state_variable("documents", documents)
                    if documents:
                        UIComponents.set_session_state_variable("processed_document_texts", [doc.get('page_content', '') for doc in documents])

                UIComponents.display_success(f"Successfully loaded pre-processed document '{uploaded_file.name}' from cache.")
            else:
                with UIComponents.display_spinner(f"Processing document '{uploaded_file.name}'... This may take a moment."):
                    try:
                        texts = pipeline.extractText(uploaded_file)
                        if texts:
                            # This returns the vector_store, but the embedder is now fitted inside the pipeline instance
                            processed_vector_store = pipeline.process_document(uploaded_file, texts)
                            
                            if processed_vector_store:
                                # Cache both the vector_store and the fitted embedder
                                data_to_cache = {
                                    'vector_store': processed_vector_store,
                                    'embedder': pipeline.embedder
                                }
                                cache_manager.save_to_cache(cache_key, data_to_cache)
                                
                                # The pipeline's vector_store is already updated by process_document
                                UIComponents.set_session_state_variable("vector_store", processed_vector_store)
                                documents = processed_vector_store.documents
                                UIComponents.set_session_state_variable("documents", documents)
                                if documents:
                                    UIComponents.set_session_state_variable("processed_document_texts", [doc.get('page_content', '') for doc in documents])
                                
                                UIComponents.display_success(f"Document '{uploaded_file.name}' processed and saved to cache.")
                            else:
                                UIComponents.display_error("Failed to process the document.")
                        else:
                            UIComponents.display_error("Failed to extract text from the document.")
                    except Exception as e:
                        UIComponents.display_error(f"An error occurred during processing: {e}")

    def get_retriever_config(self) -> dict[str, Any]:

        options, index = self.get_ui_options(option_type=constants.RetrieverType, config_name=constants.CONFIG_RETRIEVER)

        retriever_type = UIComponents.selectbox(
            "Retriever Type",
            options=options,
            index=index
        )

        top_k = UIComponents.display_slider("Top-K-Docs for Retrieval", 1, 20, 5)
        retriever_params = {}
        if retriever_type == RetrieverType.SIMILARITY.value:
            similarity_threshold = UIComponents.display_slider(constants.SIMILARITY_THRESHOLD_DISPLAY_NAME, 0.0, 1.0, 0.0, 0.01)
            retriever_params = {constants.CONFIG_SIMILARITY_THRESHOLD_PARAM: similarity_threshold, constants.CONFIG_TOP_K_PARAM: top_k}
        elif retriever_type == RetrieverType.HYBRID.value:
            keyword_weight = UIComponents.display_slider(constants.KEYWORD_WEIGHT_DISPLAY_NAME, 0.0, 1.0, 0.3, 0.05)
            retriever_params = {constants.CONFIG_KEYWORD_WEIGHT: keyword_weight, constants.CONFIG_TOP_K_PARAM: top_k}
        elif retriever_type == RetrieverType.SENTENCE_WINDOW.value:
            window_size = UIComponents.display_slider(constants.WINDOW_SIZE_DISPLAY_NAME, max_value=100, min_value=0, step=1)
            retriever_params = {constants.CONFIG_WINDOW_SIZE: window_size, constants.CONFIG_TOP_K_PARAM: top_k}
        
        retriever_config = {
            constants.CONFIG_TYPE_PARAM: retriever_type,
            constants.CONFIG_PARAM: retriever_params
        }

        return retriever_config
    
    def get_reranker_config(self) -> dict[str, Any]:
        options, index = self.get_ui_options(option_type=constants.RerankerType, config_name=constants.CONFIG_RERANKER)

        re_ranker_type = UIComponents.selectbox(
        "Re-ranker Type",
        options=options,
        index=index
        )

        top_k_for_reranking = UIComponents.display_slider("Top-K-Docs for Re-ranking", 1, 20, 5, step=1)
        re_ranker_params = {
            constants.CONFIG_TOP_K_FOR_RERANKING_PARAM: top_k_for_reranking
        }
        
        if re_ranker_type in [RerankerType.LLM.value, RerankerType.COHERE.value, RerankerType.JINA.value]:
            model = UIComponents.selectbox(
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
        UIComponents.display_divider()
        UIComponents.write(f"**{constants.CHAT_RESPONSE_CONFIG_DISPLAY_NAME}**")
        options, index = self.get_ui_options(option_type=constants.LLMServiceType, config_name=constants.CONFIG_LLM)

        llm_service = UIComponents.selectbox(
            constants.LLM_CHAT_SERVICE, 
            options=options, 
            index=index
        )
        UIComponents.set_session_state_variable("LLM_Service", llm_service)
        
        llm_model_options = self.get_llm_model_options(llm_service)

        options = list(llm_model_options.keys())
        if UIComponents.get_session_state_variable("pipeline_config").get_config(constants.CONFIG_LLM)[constants.CONFIG_TYPE_PARAM] == llm_service:
            default_option = UIComponents.get_session_state_variable("pipeline_config").get_config(constants.CONFIG_LLM)[constants.CONFIG_PARAM][constants.CONFIG_MODEL]
            default_option = self.get_name_of_llm_model_to_display(llm_service, default_option)
        else:
            default_option = options[0]
        
        # Get the index
        index = options.index(default_option)
        user_selected_llm_model = UIComponents.selectbox(
            constants.LLM_CHAT_SERVICE + " Model", 
            options=llm_model_options.keys(), 
            index=index
        )
        
        chat_response_config = {
            constants.CONFIG_TYPE_PARAM: llm_service,
            constants.CONFIG_PARAM: {
                constants.CONFIG_MODEL: llm_model_options[user_selected_llm_model]
            }
        }
        
        if UIComponents.create_button("Apply Chat Response Config", key="apply_chat_response"):
            Utils.Utils.get_pipeline().update_component(constants.CONFIG_LLM, chat_response_config)
            UIComponents.display_success("Chat response configuration updated.")

    def render_test_all_configs_section(self):
        UIComponents.create_subheader_UI("Test All Configurations")
        UIComponents.write("Click the button below to test all configurations with different combinations of chunkers, embedder, vector store, and reranker.")

        if UIComponents.create_button("Test All Configurations", key="test_all_combinations"):
            from infrastructure.Testing.RAG_Testing import test_rag_combinations
            test_rag_combinations()

    def get_chunker_config(self, chunker_type: str) -> dict:
        """Get parameters for the selected chunker type"""
        chunker_params = {}
        if chunker_type == ChunkerType.RECURSIVE.value:
            chunk_size = UIComponents.display_slider(constants.CHUNK_SIZE_DISPLAY_NAME, 10, 10000, 150)
            chunk_overlap = UIComponents.display_slider(constants.CHUNK_OVERLAP_DISPLAY_NAME, 0, 3000, 70)
            chunker_params = {
                constants.CONFIG_CHUNK_SIZE_PARAM: chunk_size,
                constants.CONFIG_CHUNK_OVERLAP_PARAM: chunk_overlap
            }
        elif chunker_type == ChunkerType.SEMANTIC.value:
            min_chunk_size = UIComponents.create_number_input(constants.MIN_CHUNK_SIZE_DISPLAY_NAME, 0, 10000, 600)
            max_chunk_size = UIComponents.create_number_input(constants.MAX_CHUNK_SIZE_DISPLAY_NAME, 0, 10000, 110)
            similarity_threshold = UIComponents.create_text_area(constants.SIMILARITY_THRESHOLD_DISPLAY_NAME, value=0.65, key="similarity_threshold_input")
            print("Similarity Threshold:", similarity_threshold)
            model_name = UIComponents.selectbox(
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
            max_sentences = UIComponents.display_slider(constants.MAX_SENTENCES_DISPLAY_NAME, 1, 20, 5)
            chunker_params = {constants.CONFIG_MAX_SENTENCES: max_sentences}
        chunker_config = {
            constants.CONFIG_TYPE_PARAM: chunker_type,
            constants.CONFIG_PARAM: chunker_params
        }
        return chunker_config

    def get_vector_store_config(self):
        """Configure vector store settings"""
        UIComponents.display_divider()

        options, index = self.get_ui_options(option_type=constants.VectorStore, config_name=constants.CONFIG_VECTOR_STORE)

        vector_store = UIComponents.selectbox(
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
        UIComponents.display_divider()
        options, index = self.get_ui_options(option_type=constants.EmbedderType, config_name=constants.CONFIG_EMBEDDER)

        embedder_type = UIComponents.selectbox(
            constants.EMBEDDER_TYPE_DISPLAY_NAME,
            options=options,
            index=index
        )
        
        if embedder_type != EmbedderType.TFIDF.value:
            emb_options = self.get_embedder_options(embedder_type)
            emb_model = UIComponents.selectbox(
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
            print("Claude Models:", [model.value for model in constants.CLAUDE_MODELS])
            return {model.display_name: model.value for model in constants.CLAUDE_MODELS}
        else:
            return {model.display_name: model.value for model in constants.GeminiLLMModel}
        
    def get_name_of_llm_model_to_display(self, llm_service: str, currModel: str) -> str:
        """Get the display name of the LLM model based on the service and model"""
        if llm_service == constants.LLMServiceType.GEMINI.value:
            for model in constants.GeminiLLMModel:
                print("Model:", currModel, "Value:", model.value)
                if model.value == currModel:
                    return model.display_name
        elif llm_service == constants.LLMServiceType.CLAUDE.value:
            for model in constants.CLAUDE_MODELS:
                print("Model:", currModel, "Value:", model.value)

                if model.value == currModel:
                    return model.display_name
        else:
            for model in constants.GeminiLLMModel:
                if model.value == model:
                    return model.display_name
    
    def get_reranker_model_options(self, reranker_type: str) -> dict:
        """Get re-ranker model options based on the selected type"""
        if reranker_type == RerankerType.LLM.value:
            return self.get_llm_model_options(UIComponents.get_session_state_variable("LLM_Service"))
        elif reranker_type == RerankerType.COHERE.value:
            return {model.value: model for model in constants.CohereLLMModel}
        elif reranker_type == RerankerType.JINA.value:
            return {model.value: model for model in constants.JINA_RERANKER_MODELS}
        return {}
    
    def apply_text_processing_config(self, chunker_params, vector_store, embedder_params):
        """Apply text processing configuration to the pipeline"""
        Utils.Utils.get_pipeline().update_component(constants.CONFIG_CHUNKER, chunker_params)
        Utils.Utils.get_pipeline().update_component(constants.CONFIG_EMBEDDER, embedder_params)
        Utils.Utils.get_pipeline().update_component(constants.CONFIG_VECTOR_STORE, vector_store)
        
        UIComponents.display_success("Text processing configuration updated.")
    
    def apply_Retrieval_and_Reranker_config(self, retriever_config, re_ranker_config):
        """Apply text processing configuration to the pipeline"""
        Utils.Utils.get_pipeline().update_component(constants.CONFIG_RETRIEVER, retriever_config)
        Utils.Utils.get_pipeline().update_component(constants.CONFIG_RERANKER, re_ranker_config)
        
        UIComponents.display_success("Retrieval and Reranking configuration updated.")