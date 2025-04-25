from enum import Enum

# Display name constants for UI labels
CHUNKER_TYPE_DISPLAY_NAME = "Chunker Type"
TEXT_PROCESSING_DISPLAY_NAME = "Text Processing"
RETRIEVAL_DISPLAY_NAME = "Retrieval"
EVALUATION_DISPLAY_NAME = "Evaluation"
EMBEDDER_TYPE_DISPLAY_NAME = "Embedder Type"

# Config Manager Names
CONFIG_CHUNKER = "chunker"
CONFIG_EMBEDDER = "embedder"
CONFIG_VECTOR_STORE = "vector_store"
CONFIG_RETRIEVER = "retriever"
CONFIG_RERANKER = "reranker"
CONFIG_LLM = "llm"
CONFIG_EVALUATOR = "evaluator"
CONFIG_TYPE_PARAM = "type"
CONFIG_PARAM = "params"
CONFIG_CHUNK_SIZE_PARAM = "chunk_size"
CONFIG_CHUNK_OVERLAP_PARAM = "chunk_overlap"
CONFIG_VECTOR_STORE_SKLEARN = "sklearn"
CONFIG_VECTOR_STORE_METRIC = "metric"
CONFIG_SIMILARITY_THRESHOLD_PARAM = "similarity_threshold"
CONFIG_TOP_K_PARAM = "top_k"
CONFIG_METRIC_COSINE = "cosine"
CONFIG_MODEL = "model"
CONFIG_CHUNKER_TYPE = "chunker_type"
CONFIG_MIN_CHUNK_SIZE_DISPLAY_NAME = "min_chunk_size"
CONFIG_MAX_CHUNK_SIZE_DISPLAY_NAME = "max_chunk_size"
CONFIG_KEYWORD_WEIGHT = "keyword_weight"
KEYWORD_WEIGHT_DISPLAY_NAME = "Keyword Weight"
CONFIG_WINDOW_SIZE = "window_size"
WINDOW_SIZE_DISPLAY_NAME = "Window Size"
CONFIG_MODEL_NAME = "model_name"
CONFIG_MAX_SENTENCES = "max_sentences"
MAX_SENTENCES_DISPLAY_NAME = "Max Sentences per Chunk"
EMBED_MODEL_DISPLAY_NAME = "Embed Model"
# constants
RERANK_EXPLAINATION = "Chunks sorted by cosine similarity scores (highest to lowest)."
VECTOR_STORE_MUST_BE_PROVIDED_ERROR_MESSAGE = "Vector store must be provided"
QUERY_TEXT_MUST_BE_PROVIDED_ERROR_MESSAGE = "Query text must be provided"
QUERY_TEXT  = "query_text"
PAGE_CONTENT = "page_content"
ID = "id"
Score = "score"
Document = "document"
SEMANTIC_SCORE = "semantic_score"
KEYWORD_SCORE = "keyword_score"
LLM_DID_NOT_SELECT_INFO_MESSAGE = "LLM did not select any chunks. Using all retrieved chunks."
QUESTION = "question"
ANSWER = "answer"
CONTEXTS = "contexts"
RERANK_EXPLANATION = "rerank_explanation"
LAST_QUERY = "last_query"
TEMP_DOCS_DIR = "temp_docs"

GEMINI_API_KEY = "GEMINI_API_KEY"
OPENAI_API_KEY = "OPEN_AI_API_KEY"
COHERE_API_KEY = "COHERE_API_KEY"
VOYAGE_API_KEY = "VOYAGE_API_KEY"

FAITHFULNESS = "faithfulness"
ANSWER_CORRECTNESS = "answer_correctness"
CONTEXT_PRECISION = "context_precision"
CONTEXT_RECALL = "context_recall"
ANSWER_RELEVANCY = "answer_relevancy"

# Model Names
SENTENCE_TRANSFORMER_MODEL_ALL_MINI = "all-MiniLM-L6-v2"
SENTENCE_TRANSFORMER_MODEL_PARAPHRASE_MINI = "paraphrase-MiniLM-L3-v2"
COHERE_EMBED_MODEL_DEFAULT = "embed-v4.0"
COHERE_EMBED_MODEL_ENG = "embed-english-v3.0"

MIN_CHUNK_SIZE_DISPLAY_NAME = "Min Chunk Size"
MAX_CHUNK_SIZE_DISPLAY_NAME = "Max Chunk Size"
CHUNK_SIZE_DISPLAY_NAME = "Chunk Size"
CHUNK_OVERLAP_DISPLAY_NAME = "Chunk Overlap"
MODEL_NAME_DISPLAY_NAME = "Model Name"
SIMILARITY_THRESHOLD_DISPLAY_NAME = "Similarity Threshold"
METADATA = "metadata"
REFERENCE = "reference"
# extensions
PDF_EXTENSION = ".pdf"
DOCX_EXTENSION = ".docx"
TXT_EXTENSION = ".txt"
CSV_EXTENSION = ".csv"

LLM_MODEL_OPTIONS = "LLM_Model_Options"

GROUND_TRUTH_DISPLAY_NAME = "Ground Truth"
GROUND_TRUTH_DEFAULT_VALUE = "It sends large blocks of data continuously without start and stop bits. Requires clock synchronization between sender and receiver and is more efficient."

# Component type enums with string values matching config and UI
class ChunkerType(Enum):
    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"

class EmbedderType(Enum):
    TFIDF = "tfidf"
    GEMINI = "gemini"
    COHERE = "cohere"
    VOYAGE = "Voyage"

class RetrieverType(Enum):
    SIMILARITY = "similarity"
    HYBRID = "hybrid"
    SENTENCE_WINDOW = "sentence_window"

class RerankerType(Enum):
    COSINE = "cosine"
    LLM = "llm"
    COHERE = "cohere"

class EvaluatorType(Enum):
    SIMPLE = "simple"
    RAGAS = "ragas"

class LLMServiceType(Enum):
    GEMINI = "gemini"
    COHERE = "cohere"

# Common LLM model names
class GeminiLLMModel(Enum):
    GEMINI_FLASH = "gemini-2.0-flash"
    GEMINI_PRO = "gemini-2.5-pro"

# Common LLM model names
class CohereLLMModel(Enum):
    RERANK_DEFAULT_MODEL = "rerank-v3.5"
    RERANK_ENGLISH = "rerank-english-v3.0"
    RERANK_MULTLINGUAL = "rerank-multilingual-v3.0"

class VoyageEmbedModels(Enum):
    VOYAGE_EMBED_DEFAULT_MODEL = "voyage-3-large"
    VOYAGE_3_EMBED_MODEL = "voyage-3"
    VOYAGE_3_LITE_EMBED_MODEL = "voyage-3-lite"
    VOYAGE_CODE_2_EMBED_MODEL = "voyage-code-2"