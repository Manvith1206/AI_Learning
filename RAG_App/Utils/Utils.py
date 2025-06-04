from UI.UI_Components import UIComponents
from infrastructure.Common.rag_pipeline import RAGPipeline

@staticmethod
def get_pipeline() -> RAGPipeline:
        """Get or initialize the pipeline"""
        
        if not UIComponents.get_session_state_variable("pipeline_created", False):
            with UIComponents.display_spinner("Initializing RAG pipeline..."):
                UIComponents.set_session_state_variable(var_name='pipeline', value=RAGPipeline(UIComponents.get_session_state_variable('pipeline_config')))
                UIComponents.set_session_state_variable(var_name="pipeline_created",value=True)
        return UIComponents.get_session_state_variable("pipeline", None)