from UI.ui_components import UIComponents
import os

from infrastructure.common.rag_pipeline import RAGPipeline
import infrastructure.common.rag_constants as constants

@staticmethod
def get_pipeline() -> RAGPipeline:
    """Get or initialize the pipeline"""

    if not UIComponents.get_session_state_variable(constants.Constants.PIPELINE_CREATED, False):
        with UIComponents.display_spinner("Initializing RAG pipeline..."):
            os.environ["OPENAI_API_KEY"] = get_env_var(constants.APIKeys.OPENAI_API_KEY)
            rag_pipeline = RAGPipeline(
                geminiApiKey=get_env_var(constants.APIKeys.GEMINI_API_KEY),
                cohereApiKey=get_env_var(constants.APIKeys.COHERE_API_KEY),
                voyageApiKey=get_env_var(constants.APIKeys.VOYAGE_API_KEY),
                mistralApiKey=get_env_var(constants.APIKeys.MISTRAL_API_KEY),
                pineconeApiKey=get_env_var(constants.APIKeys.PINECONE_API_KEY),
                jinaApiKey=get_env_var(constants.APIKeys.JINA_RERANKER_API_KEY),
                claudeApiKey=get_env_var(constants.APIKeys.CLAUDE_API_KEY),
                config_manager=UIComponents.get_session_state_variable(constants.Constants.PIPELINE_CONFIG),
                warning_callback=handleWarning, 
                error_callback=handleError,
                process_doc_callback=process_doc_success,
                vector_store=None)
            
            UIComponents.set_session_state_variable(var_name=constants.Constants.PIPELINE, value=rag_pipeline)
            UIComponents.set_session_state_variable(var_name=constants.Constants.PIPELINE_CREATED,value=True)
    return UIComponents.get_session_state_variable(constants.Constants.PIPELINE, None)

@staticmethod
def get_env_var(var_name: str):
    return os.getenv(var_name)

@staticmethod
def handleWarning(message: str):
    UIComponents.display_warning(message=message)

@staticmethod 
def process_doc_success(message: str):
    UIComponents.display_success(message=message)

@staticmethod
def handleError(message: str):
    UIComponents.display_error(message=message)