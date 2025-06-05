import os
import sys

# Add project root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

# Override built-in print to log outputs to file
import builtins
LOG_FILE_PATH = os.path.join(ROOT, "test_results_v2.log")
LOG_FILE = open(LOG_FILE_PATH, "w", encoding="utf-8")
_original_print = builtins.print
# def print(*args, **kwargs):
#     _original_print(*args, **kwargs)
#     try:
#         LOG_FILE.write(" ".join(str(a) for a in args) + "\n")
#         LOG_FILE.flush()
#     except Exception:
#         pass

from infrastructure.Common.rag_pipeline import RAGPipeline
from config import ConfigManager
import infrastructure.Common.RAG_Constants as constants

# Path to the test document
TEST_FILE_PATH = r"E:\Manvith\Coding\AI\RAG\RAG_App.infrastructure\TestFile\DCA2104 Unit-08_V1.1.pdf"
TEST_QUESTION = "What is Synchronous Transmission"

# Load test questions for evaluation
QUESTIONS_FILE = os.path.join(ROOT, "generated_questions.text")
with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    TEST_QUESTIONS = [line.strip() for line in f if line.strip()]

class DummyFile:
    """
    Wrapper to mimic Streamlit UploadedFile interface for local tests
    """
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)

    def getbuffer(self):
        with open(self.path, 'rb') as f:
            return f.read()


# Parameter grids for automated testing of types with params
CHUNKER_PARAM_GRID = [
    (constants.ChunkerType.RECURSIVE.value, {constants.CONFIG_CHUNK_SIZE_PARAM: 600, constants.CONFIG_CHUNK_OVERLAP_PARAM: 200}),
    (constants.ChunkerType.SEMANTIC.value, {constants.CONFIG_MIN_CHUNK_SIZE_DISPLAY_NAME: 600, constants.CONFIG_MAX_CHUNK_SIZE_DISPLAY_NAME: 1000, constants.CONFIG_SIMILARITY_THRESHOLD_PARAM: 0.65, constants.CONFIG_MODEL_NAME: constants.SENTENCE_TRANSFORMER_MODEL_ALL_MINI})
]

EMBEDDER_PARAM_GRID = [(constants.EmbedderType.GEMINI.value, {constants.CONFIG_MODEL: m.value}) for m in constants.GeminiEmbedModels] \
  + [(constants.EmbedderType.COHERE.value, {constants.CONFIG_MODEL: m.value}) for m in constants.CohereEmbedModels] \
  + [(constants.EmbedderType.VOYAGE.value, {constants.CONFIG_MODEL: m.value}) for m in constants.VoyageEmbedModels]

VECTOR_STORE_PARAM_GRID = [
    (constants.CONFIG_VECTOR_STORE_SKLEARN, {constants.CONFIG_VECTOR_STORE_METRIC: constants.CONFIG_METRIC_COSINE}),
    (constants.CONFIG_VECTOR_STORE_FAISS, {}),
]

RETRIEVER_PARAM_GRID = [
    (constants.RetrieverType.SIMILARITY.value, {constants.CONFIG_SIMILARITY_THRESHOLD_PARAM: 0.0}),
    (constants.RetrieverType.HYBRID.value, {constants.CONFIG_KEYWORD_WEIGHT: 0.3}),
]

RERANKER_PARAM_GRID = [
    (constants.RerankerType.COSINE.value, {}),
] + [(constants.RerankerType.LLM.value, {constants.CONFIG_PARAM: constants.GeminiLLMModel.GEMINI_FLASH.value})] \
  + [(constants.RerankerType.COHERE.value, {constants.CONFIG_PARAM: constants.CohereLLMModel.RERANK_DEFAULT_MODEL.value})]

EVALUATOR_PARAM_GRID = [
    (constants.EvaluatorType.RAGAS.value, {}),
]

# Rewrite test_all_combinations to iterate over type+param combos
def test_all_combinations():
    results = {}
    for chunker_type, chunker_params in CHUNKER_PARAM_GRID:
        for embedder_type, embedder_params in EMBEDDER_PARAM_GRID:
            for vs_type, vs_params in VECTOR_STORE_PARAM_GRID:
                for retriever_type, retriever_params in RETRIEVER_PARAM_GRID:
                    for reranker_type, reranker_params in RERANKER_PARAM_GRID:

                        key = "|".join([chunker_type, embedder_type, vs_type, retriever_type, reranker_type])
                        print("---------------------")
                        print("Combinations: " + key)
                        print("Chunker: " + chunker_type, "Params: " + str(chunker_params))
                        print("Embedder: " + embedder_type, "Params: " + str(embedder_params))
                        print("VEctor store: " + vs_type, "Params: " + str(vs_params))
                        print("Retriever: " + retriever_type, "Params: " + str(retriever_params))
                        
                        print("Retriever: " + reranker_type, "Params: " + str(reranker_params))
                        cm = ConfigManager()
                        pipeline = RAGPipeline(cm)
                        pipeline.update_component(constants.CONFIG_CHUNKER, {constants.CONFIG_TYPE_PARAM: chunker_type, constants.CONFIG_PARAM: chunker_params})
                        pipeline.update_component(constants.CONFIG_EMBEDDER, {constants.CONFIG_TYPE_PARAM: embedder_type, constants.CONFIG_PARAM: embedder_params})
                        vs_cfg = {constants.CONFIG_TYPE_PARAM: vs_type}
                        if vs_type == constants.CONFIG_VECTOR_STORE_SKLEARN:
                            vs_cfg[constants.CONFIG_VECTOR_STORE_METRIC] = constants.CONFIG_METRIC_COSINE
                        pipeline.update_component(constants.CONFIG_VECTOR_STORE, vs_cfg)
                        pipeline.update_component(constants.CONFIG_RETRIEVER, {constants.CONFIG_TYPE_PARAM: retriever_type, constants.CONFIG_PARAM: retriever_params, constants.CONFIG_TOP_K_PARAM: 5})
                        pipeline.update_component(constants.CONFIG_RERANKER, {constants.CONFIG_TYPE_PARAM: reranker_type, constants.CONFIG_PARAM: reranker_params})
                        try:
                            docs, chunks = pipeline.process_document(DummyFile(TEST_FILE_PATH))
                            status = "success" if docs and chunks else "no_output"
                        except Exception as e:
                            status = f"error_process: {e}"
                            
                            print(f"Error processing: {e}")
                        metrics_summary = {}
                        if status == "success":
                            agg = {}
                            pipeline.query(str(TEST_QUESTION))
                            
                        results[key] = {"status": status}
    total = len(results)
    successes = sum(1 for v in results.values() if v["status"] == "success")
    print(f"\nTested {total} combos; successes: {successes}")
    if successes > 0:
        best = max((k for k, v in results.items() if v["status"] == "success"), key=lambda k: sum(results[k]["metrics"].values()))
        print(f"Best config: {best} -> metrics: {results[best]['metrics']}")
    assert successes > 0, "No successful configurations found"



