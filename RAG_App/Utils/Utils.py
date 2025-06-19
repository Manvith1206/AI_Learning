from UI.UI_Components import UIComponents
import os

from infrastructure.Common.rag_pipeline import RAGPipeline
import infrastructure.Common.RAG_Constants as constants

@staticmethod
def get_pipeline() -> RAGPipeline:
    """Get or initialize the pipeline"""
    
    if not UIComponents.get_session_state_variable("pipeline_created", False):
        with UIComponents.display_spinner("Initializing RAG pipeline..."):
            
            os.environ["OPENAI_API_KEY"] = get_env_var(constants.OPENAI_API_KEY)
            rag_pipeline = RAGPipeline(
                geminiApiKey=get_env_var(constants.GEMINI_API_KEY),
                cohereApiKey=get_env_var(constants.COHERE_API_KEY),
                voyageApiKey=get_env_var(constants.VOYAGE_API_KEY),
                mistralApiKey=get_env_var(constants.MISTRAL_API_KEY),
                pineconeApiKey=get_env_var(constants.PINECONE_API_KEY),
                jinaApiKey=get_env_var(constants.JINA_RERANKER_API_KEY),
                claudeApiKey=get_env_var(constants.CLAUDE_API_KEY),
                config_manager=UIComponents.get_session_state_variable('pipeline_config'),
                warning_callback=handleWarning, 
                error_callback=handleError,
                process_doc_callback=process_doc_success)
            
            UIComponents.set_session_state_variable(var_name='pipeline', value=rag_pipeline)
            UIComponents.set_session_state_variable(var_name="pipeline_created",value=True)
    return UIComponents.get_session_state_variable("pipeline", None)

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