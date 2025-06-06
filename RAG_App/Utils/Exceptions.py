
class RAGException(Exception):
    """Base exception for all RAG application exceptions"""
    pass

class ConfigurationError(RAGException):
    """Exception raised for errors in the configuration"""
    pass


class ComponentInitializationError(RAGException):
    """Exception raised when a component fails to initialize"""
    pass


class DocumentProcessingError(RAGException):
    """Exception raised when document processing fails"""
    pass


class EmbeddingError(RAGException):
    """Exception raised when embedding fails"""
    pass


class VectorStoreError(RAGException):
    """Exception raised when vector store operations fail"""
    pass


class RetrievalError(RAGException):
    """Exception raised when retrieval fails"""
    pass


class RerankingError(RAGException):
    """Exception raised when reranking fails"""
    pass


class LLMServiceError(RAGException):
    """Exception raised when LLM service fails"""
    pass


class EvaluationError(RAGException):
    """Exception raised when evaluation fails"""
    pass

class APIKeyError(RAGException):
    """Exception raised when API key is not found"""
    pass