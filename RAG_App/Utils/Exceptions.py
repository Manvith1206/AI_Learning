class RAGException(Exception):
    """Base exception for all RAG application exceptions"""
    pass
class FlashcardGenerationError(RAGException):
    pass
class PipelineError(RAGException):
    pass
class MissingGeminiAPIKeyError(RAGException):
    pass
class CustomEvalError(RAGException):
    pass
class ExtractionOfText(RAGException):
    pass
class DocumentProcessError(RAGException):
    pass
class ErrorDuringQuery(RAGException):
    pass
class EvaluationError(RAGException):
    pass