class PipelineError(Exception):
    """Base class for exceptions in the RAG pipeline."""
    pass

class ComponentBuildError(PipelineError):
    """Raised when a pipeline component fails to build."""
    pass

class MissingConfigurationError(PipelineError):
    """Raised when a required configuration is missing."""
    pass

class InvalidConfigurationError(PipelineError):
    """Raised when a configuration is invalid."""
    pass

class EvaluationError(PipelineError):
    """Raised when an error occurs during the evaluation process."""
    pass

class FlashcardGenerationError(PipelineError):
    """Raised when an error occurs during flashcard generation."""
    pass

class DocumentProcessingError(PipelineError):
    """Raised when an error occurs during document processing (loading, splitting)."""
    pass
