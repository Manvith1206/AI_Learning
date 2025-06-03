# Display name constants for UI labels (subset, to be expanded)
CHUNKER_TYPE_DISPLAY_NAME = "Chunker Type"
EMBEDDER_TYPE_DISPLAY_NAME = "Embedder Type"
VECTOR_STORE_DISPLAY_NAME = "Vector Store"
RETRIEVAL_DISPLAY_NAME = "Retrieval"
EVALUATION_DISPLAY_NAME = "Evaluation"
CHAT_RESPONSE_CONFIG_DISPLAY_NAME = "Chat Interface"
TEXT_PROCESSING_DISPLAY_NAME = "Text Processing"

# Common messages
UNABLE_TO_RETRIEVE_MESSAGE = "Unable to retrieve documents currently. Try Again after some time."
LLM_DID_NOT_SELECT_INFO_MESSAGE = "LLM did not select any chunks. Using all retrieved chunks."

# Common field names / keys
PAGE_CONTENT = "page_content"
ID = "id"
SCORE = "score"
DOCUMENT = "document"
METADATA = "metadata"
REFERENCE = "reference"
QUESTION = "question"
ANSWER = "answer"
CONTEXTS = "contexts"

# File extensions
PDF_EXTENSION = ".pdf"
DOCX_EXTENSION = ".docx"
TXT_EXTENSION = ".txt"
CSV_EXTENSION = ".csv"

# Add other constants from RAG_Constants.py and Constants.py as appropriate in later focused steps.
# For example, the extensive Enums from RAG_Constants.py will likely go into app/models/enums.py or similar.
# Config keys like CONFIG_CHUNKER will be evaluated in context of service/component refactoring.
