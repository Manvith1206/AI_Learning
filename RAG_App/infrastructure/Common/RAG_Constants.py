from enum import Enum

# Display name constants for UI labels
CHUNKER_TYPE_DISPLAY_NAME = "Chunker Type"
TEXT_PROCESSING_DISPLAY_NAME = "Text Processing"
RETRIEVAL_DISPLAY_NAME = "Retrieval"
EVALUATION_DISPLAY_NAME = "Evaluation"
CHAT_RESPONSE_CONFIG_DISPLAY_NAME = "Chat Interface"

EMBEDDER_TYPE_DISPLAY_NAME = "Embedder Type"
VECTOR_STORE_DISPLAY_NAME = "Vector Store"

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
CONFIG_VECTOR_STORE_PINCONE = "pinecone"

CONFIG_BATCH_SIZE = "batch_size"
BATCH_SIZE_DISPLAY_NAME = "Batch Size"

CONFIG_VECTOR_STORE_FAISS = "faiss"
CONFIG_VECTOR_STORE_METRIC = "metric"
CONFIG_SIMILARITY_THRESHOLD_PARAM = "similarity_threshold"
CONFIG_TOP_K_PARAM = "top_k"
CONFIG_TOP_K_FOR_RERANKING_PARAM = "top_k_for_reranking"
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
CONFIG_CHAT_RESPONSE = "chat_response"

# constants
LOADING_DISPLAY_MESSAGE_FOR_INITIALZING_PAGE = "Initializing Page"
LOADING_DISPLAY_MESSAGE_FOR_MAIN_PAGE = "Initialing Main Page"

COSINE_SIMILARITY_RERANK_EXPLAINATION = "Chunks sorted by cosine similarity scores (highest to lowest)."
NO_EXPLAINATION_NEEDED_MESSAGE = "No explanation needed."
UNABLE_TO_RETRIEVE_MESSAGE = "Unable to retrieve documents currently. Try Again after some time."
NUM_OF_FLASHCARDS = 5

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

PINE_CONE_INDEX_NAME = "test-rag-v1"
CHROMA_COLLECTION_NAME = "chroma-rag-v1"

GEMINI_API_KEY = "GEMINI_API_KEY"
OPENAI_API_KEY = "OPEN_AI_API_KEY"
COHERE_API_KEY = "COHERE_API_KEY"
VOYAGE_API_KEY = "VOYAGE_API_KEY"
PINECONE_API_KEY = "PINECONE_API_KEY"
JINA_RERANKER_API_KEY = "JINA_RERANKER_API_KEY"
MISTRAL_API_KEY = "MISTRAL_API_KEY"
CLAUDE_API_KEY = "CLAUDE_API_KEY"

# Metric Names for ragas
FAITHFULNESS = "faithfulness"
ANSWER_CORRECTNESS = "answer_correctness"
CONTEXT_PRECISION = "context_precision"
CONTEXT_RECALL = "context_recall"
ANSWER_RELEVANCY = "answer_relevancy"


# Metric Names for deep eval
DEEP_EVAL_FAITHFULNESS = "Faithfulness"
DEEP_EVAL_ANSWER_CORRECTNESS = "Answer Correctness"
DEEP_EVAL_CONTEXT_PRECISION = "Contextual Precision"
DEEP_EVAL_CONTEXT_RECALL = "Contextual Recall"
DEEP_EVAL_ANSWER_RELEVANCY = "Answer Relevancy"

# Model Names
SENTENCE_TRANSFORMER_MODEL_ALL_MINI = "all-MiniLM-L6-v2"
SENTENCE_TRANSFORMER_MODEL_PARAPHRASE_MINI = "paraphrase-MiniLM-L3-v2"

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

GROUND_TRUTH_CONFIG_NAME = "ground_truth"
RESPONSE_CONFIG_NAME = "response"
CONTEXTS_CONFIG_NAME = "contexts"
STATUS_CONFIG_NAME = "status"
ERROR_CONFIG_NAME = "error"
METRICS_CONFIG_NAME = "metrics"

MISTRAL_EMBED_MODEL = "mistral-embed"

LLM_CHAT_SERVICE = "LLM Chat Service"
# Component type enums with string values matching config and UI
class ChunkerType(Enum):
    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    PAGE = "page"
    SEMANTIC_WITH_LANGCHAIN = "semantic_with_langchain"

class EmbedderType(Enum):
    TFIDF = "tfidf"
    GEMINI = "gemini"
    COHERE = "cohere"
    VOYAGE = "Voyage"
    MISTRAL = "mistral"

class RetrieverType(Enum):
    SIMILARITY = "similarity"
    HYBRID = "hybrid"
    SENTENCE_WINDOW = "sentence_window"

class RerankerType(Enum):
    COSINE = "Cosine"
    LLM = "LLM"
    COHERE = "Cohere"
    JINA = 'Jina'

class EvaluatorType(Enum):
    SIMPLE = "Simple"
    RAGAS = "Ragas"
    CUSTOM = "Custom"
    DEEP_EVAL = "DeepEval"

class LLMServiceType(Enum):
    GEMINI = "Gemini"
    # COHERE = "cohere"
    CLAUDE = "Claude"

# Common LLM model names
class GeminiLLMModel(Enum):
    GEMINI_FLASH = "gemini-2.0-flash"
    GEMINI_PRO = "gemini-2.5-pro-preview-05-06"
    GEMINI_TWO_5_FLASH = "models/gemini-2.5-flash-preview-05-20"
    
    @property
    def display_name(self):
        # Optional: custom formatting
        return self.name.replace("_", " ").title()
# vector stores
class VectorStore(Enum):
    SCIKIT_LEARN = "sckit_learn"
    FAISS = "faiss"
    PINE_CONE = "pine-cone"
    CHROMA = "chroma"

# Common LLM model names
class CohereRerankingModels(Enum):
    RERANK_DEFAULT_MODEL = "rerank-v3.5"
    RERANK_ENGLISH = "rerank-english-v3.0"
    RERANK_MULTLINGUAL = "rerank-multilingual-v3.0"

class CohereEmbedModels(Enum):
    COHERE_EMBED_MODEL_DEFAULT = "embed-v4.0"
    COHERE_EMBED_MODEL_ENG = "embed-english-v3.0"
    COHERE_EMBEDDING_MULTILINGUAL_V3_0 = "embed-multilingual-v3.0"


class VoyageEmbedModels(Enum):
    VOYAGE_EMBED_DEFAULT_MODEL = "voyage-3-large"
    VOYAGE_3_EMBED_MODEL = "voyage-3"
    VOYAGE_3_LITE_EMBED_MODEL = "voyage-3-lite"
    VOYAGE_CODE_2_EMBED_MODEL = "voyage-code-2"

class GeminiEmbedModels(Enum):
    GEMINI_EMBED_EXP_MODEL =  "gemini-embedding-exp-03-07"
    GEMINI_TEXT_EMBED_MODEL = "models/text-embedding-004"
    GEMINI_EMBED_001_MODEL = "models/embedding-001"

class JINA_RERANKER_MODELS(Enum):
    JINA_RERANKER_V1_TURBO = "jina-reranker-v1-turbo-en"
    JINA_RERANKER_TINY = "jina-reranker-v1-tiny-en"
    JINA_RERANKER_M0 = "jina-reranker-m0"
    JINA_RERANKER_MULTILINGUAL = "jina-reranker-v2-base-multilingual"
class MISTRAL_EMBED_MODELS(Enum):
    MISTRAL_EMBED_MODEL_DEFAULT = "mistral-embed"

class CLAUDE_MODELS(Enum):
    CLAUDE_SONNET_THREE_7 = "claude-3-7-sonnet-20250219"
    CLAUDE_SONNET_THREE_5 = "claude-3-5-sonnet-20241022"
    CLAUDE_HAIKU_THREE_5 = "claude-3-5-haiku-20241022"
    CLAUDE_OPUS_THREE = "claude-3-opus-20240229"
    CLAUDE_OPUS_FOUR = "claude-opus-4-20250514"
    CLAUDE_SONNET_FOUR = "claude-sonnet-4-20250514"
    
    @property
    def display_name(self):
        # Optional: custom formatting
        return self.name.replace("_", " ").title()
    