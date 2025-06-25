class RAGException(Exception):
    """Base exception for all RAG application exceptions."""
    pass


class MissingConfigurationError(RAGException):
    """Raised when a required configuration or API key is missing."""
    pass


class ComponentBuildError(RAGException):
    """Raised when a component fails to build."""
    pass


class InvalidConfigurationError(RAGException):
    """Raised when a configuration is invalid."""
    pass


class PipelineError(RAGException):
    """Raised when a pipeline operation fails."""
    pass


class DocumentProcessingError(RAGException):
    """Raised during an error in document processing."""
    pass


class ExtractionOfTextError(DocumentProcessingError):
    """Raised during an error in text extraction from a document."""
    pass


class QueryProcessingError(RAGException):
    """Raised during an error in query processing."""
    pass


class EvaluationError(RAGException):
    """Raised during an error in evaluation."""
    pass


class CustomEvalError(EvaluationError):
    """Raised during a custom evaluation error."""
    pass


class FlashcardGenerationError(RAGException):
    """Raised during an error in flashcard generation."""
    pass
