from UI.UI_Components import UIComponents
from infrastructure.Common.rag_pipeline import RAGPipeline
import os

@staticmethod
def get_pipeline() -> RAGPipeline:
        """Get or initialize the pipeline"""
        
        if not UIComponents.get_session_state_variable("pipeline_created", False):
            with UIComponents.display_spinner("Initializing RAG pipeline..."):
                rag_pipeline = RAGPipeline(
                    config_manager=UIComponents.get_session_state_variable('pipeline_config'),
                    warning_callback=handleWarning, 
                    error_callback=handleError)
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
def handleError(message: str):
     UIComponents.display_error(message=message)