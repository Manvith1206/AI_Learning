import os
import sys
import csv
import time
from datetime import datetime
import pandas as pd

# Add project root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from rag_modular.rag_pipeline import RAGPipeline
from rag_modular.config_manager import ConfigManager
import rag_modular.RAG_Constants as constants

# Path to the test document and output file
TEST_FILE_PATH = r"E:\Manvith\Coding\AI\RAG\rag_modular\TestFile\DCA2104 Unit-08_V1.1.pdf"
RESULTS_CSV_PATH = os.path.join(ROOT, "rag_evaluation_results.csv")

# Test query to use for all combinations
TEST_QUERIES = [
    "What is Synchronous Transmission?",
    "Explain the difference between synchronous and asynchronous transmission.",
    "What are the advantages of synchronous transmission?"
]

# Ground truth answers (optional - if available)
GROUND_TRUTHS = {
    "What is Synchronous Transmission?": "",  # Add ground truth if available
    "Explain the difference between synchronous and asynchronous transmission.": "",
    "What are the advantages of synchronous transmission?": ""
}

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

# Parameter grids for automated testing
CHUNKER_PARAM_GRID = [
    (constants.ChunkerType.RECURSIVE.value, {
        constants.CONFIG_TYPE_PARAM: constants.ChunkerType.RECURSIVE.value, 
        constants.CONFIG_PARAM: {
            constants.CONFIG_CHUNK_SIZE_PARAM: 600, 
            constants.CONFIG_CHUNK_OVERLAP_PARAM: 200
        }
    }),
    (constants.ChunkerType.SEMANTIC.value, {
        constants.CONFIG_TYPE_PARAM: constants.ChunkerType.SEMANTIC.value, 
        constants.CONFIG_PARAM: {
            constants.CONFIG_MIN_CHUNK_SIZE_DISPLAY_NAME: 600, 
            constants.CONFIG_MAX_CHUNK_SIZE_DISPLAY_NAME: 1000, 
            constants.CONFIG_SIMILARITY_THRESHOLD_PARAM: 0.65, 
            constants.CONFIG_MODEL_NAME: constants.SENTENCE_TRANSFORMER_MODEL_ALL_MINI
        }
    }),
    (constants.ChunkerType.SENTENCE.value, {
        constants.CONFIG_TYPE_PARAM: constants.ChunkerType.SENTENCE.value, 
        constants.CONFIG_PARAM: {
            constants.CONFIG_MAX_SENTENCES: 5
        }
    })
]

EMBEDDER_PARAM_GRID = [
    (constants.EmbedderType.TFIDF.value, {
        constants.CONFIG_TYPE_PARAM: constants.EmbedderType.TFIDF.value
    }),
    (constants.EmbedderType.GEMINI.value, {
        constants.CONFIG_TYPE_PARAM: constants.EmbedderType.GEMINI.value, 
        constants.CONFIG_MODEL: constants.GeminiEmbedModels.GEMINI_TEXT_EMBED_MODEL.value,
        constants.CONFIG_BATCH_SIZE: 0
    }),
    (constants.EmbedderType.COHERE.value, {
        constants.CONFIG_TYPE_PARAM: constants.EmbedderType.COHERE.value, 
        constants.CONFIG_MODEL: constants.CohereEmbedModels.COHERE_EMBED_MODEL_DEFAULT.value,
        constants.CONFIG_BATCH_SIZE: 0
    }),
    (constants.EmbedderType.VOYAGE.value, {
        constants.CONFIG_TYPE_PARAM: constants.EmbedderType.VOYAGE.value, 
        constants.CONFIG_MODEL: constants.VoyageEmbedModels.VOYAGE_EMBED_DEFAULT_MODEL.value,
        constants.CONFIG_BATCH_SIZE: 0
    }),
    (constants.EmbedderType.MISTRAL.value, {
        constants.CONFIG_TYPE_PARAM: constants.EmbedderType.MISTRAL.value, 
        constants.CONFIG_MODEL: constants.MISTRAL_EMBED_MODELS.MISTRAL_EMBED_DEFAULT_MODEL.value,
        constants.CONFIG_BATCH_SIZE: 10
    })
]

VECTOR_STORE_PARAM_GRID = [
    (constants.VectorStore.SCIKIT_LEARN.value, {
        constants.CONFIG_TYPE_PARAM: constants.CONFIG_VECTOR_STORE_SKLEARN, 
        constants.CONFIG_VECTOR_STORE_METRIC: constants.CONFIG_METRIC_COSINE
    }),
    (constants.VectorStore.FAISS.value, {
        constants.CONFIG_TYPE_PARAM: constants.CONFIG_VECTOR_STORE_FAISS
    }),
    (constants.VectorStore.PINE_CONE.value, {
        constants.CONFIG_TYPE_PARAM: constants.CONFIG_VECTOR_STORE_PINCONE
    })
]

RETRIEVER_PARAM_GRID = [
    (constants.RetrieverType.SIMILARITY.value, {
        constants.CONFIG_TYPE_PARAM: constants.RetrieverType.SIMILARITY.value, 
        constants.CONFIG_PARAM: {
            constants.CONFIG_SIMILARITY_THRESHOLD_PARAM: 0.0
        },
        constants.CONFIG_TOP_K_PARAM: 5
    }),
    (constants.RetrieverType.HYBRID.value, {
        constants.CONFIG_TYPE_PARAM: constants.RetrieverType.HYBRID.value, 
        constants.CONFIG_PARAM: {
            constants.CONFIG_KEYWORD_WEIGHT: 0.3
        },
        constants.CONFIG_TOP_K_PARAM: 5
     }),
    # (constants.RetrieverType.SENTENCE_WINDOW.value, {
    #     constants.CONFIG_TYPE_PARAM: constants.RetrieverType.SENTENCE_WINDOW.value, 
    #     constants.CONFIG_PARAM: {
    #         constants.CONFIG_WINDOW_SIZE: 3
    #     },
    #     constants.CONFIG_TOP_K_PARAM: 5
    # })
]

RERANKER_PARAM_GRID = [
    (constants.RerankerType.COSINE.value, {
        constants.CONFIG_TYPE_PARAM: constants.RerankerType.COSINE.value, 
        constants.CONFIG_PARAM: {}
    }),
    (constants.RerankerType.LLM.value, {
        constants.CONFIG_TYPE_PARAM: constants.RerankerType.LLM.value, 
        constants.CONFIG_PARAM: constants.GeminiLLMModel.GEMINI_FLASH.value
    }),
    (constants.RerankerType.COHERE.value, {
        constants.CONFIG_TYPE_PARAM: constants.RerankerType.COHERE.value, 
        constants.CONFIG_PARAM: constants.CohereLLMModel.RERANK_DEFAULT_MODEL.value
    }),
    (constants.RerankerType.JINA.value, {
        constants.CONFIG_TYPE_PARAM: constants.RerankerType.JINA.value, 
        constants.CONFIG_PARAM: constants.JINA_RERANKER_MODELS.JINA_RERANKER_DEFAULT_MODEL.value
    })
]

def evaluate_rag_combination(
    chunker_config, 
    embedder_config, 
    vector_store_config, 
    retriever_config, 
    reranker_config, 
    query
):
    """
    Evaluate a specific RAG combination with the given query
    
    Returns:
        dict: Results including success/failure status and metrics if successful
    """
    try:
        # Initialize pipeline with the specific configuration
        cm = ConfigManager()
        pipeline = RAGPipeline(cm)
        
        # Update components with the specific configurations
        pipeline.update_component(constants.CONFIG_CHUNKER, chunker_config)
        pipeline.update_component(constants.CONFIG_EMBEDDER, embedder_config)
        pipeline.update_component(constants.CONFIG_VECTOR_STORE, vector_store_config)
        pipeline.update_component(constants.CONFIG_RETRIEVER, retriever_config)
        pipeline.update_component(constants.CONFIG_RERANKER, reranker_config)
        
        # Process document
        print(f"Processing document with configuration...")
        start_time = time.time()
        docs, chunks = pipeline.process_document(DummyFile(TEST_FILE_PATH))
        process_time = time.time() - start_time
        
        if not docs or not chunks:
            return {
                "status": "failed",
                "error": "No documents or chunks produced",
                "process_time": process_time
            }
        
        # Run query
        print(f"Running query: {query}")
        query_start_time = time.time()
        response = pipeline.query(query)
        query_time = time.time() - query_start_time
        
        # Get evaluation metrics if available
        try:
            ground_truth = GROUND_TRUTHS.get(query, "")
            metrics = pipeline.evaluate(
                question=query, 
                answer=response[constants.ANSWER], 
                contexts=response[constants.CONTEXTS], 
                ground_truths=[ground_truth] if ground_truth else None
            )
        except Exception as e:
            metrics = {"error_evaluating": str(e)}
        
        # Manual scoring placeholders (would be filled by human evaluators)
        manual_metrics = {
            "faithfulness": "",  # Score 1-5
            "context_relevancy": "",  # Score 1-5
            "context_recall": "",  # Score 1-5
            "answer_correctness": "",  # Score 1-5
            "answer_relevancy": "",  # Score 1-5
            "overall_score": ""  # Score 1-5
        }
        
        return {
            "status": "success",
            "response": response[constants.ANSWER],
            "contexts": response[constants.CONTEXTS],
            "process_time": process_time,
            "query_time": query_time,
            "metrics": metrics,
            "manual_metrics": manual_metrics
        }
    
    except Exception as e:
        print(f"Error evaluating combination: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

def run_evaluation():
    """
    Run evaluation for all combinations and save results to CSV
    """
    results = []
    total_combinations = (
        len(CHUNKER_PARAM_GRID) * 
        len(EMBEDDER_PARAM_GRID) * 
        len(VECTOR_STORE_PARAM_GRID) * 
        len(RETRIEVER_PARAM_GRID) * 
        len(RERANKER_PARAM_GRID) *
        len(TEST_QUERIES)
    )
    
    print(f"Starting evaluation of {total_combinations} combinations...")
    
    # Create CSV file with headers
    with open(RESULTS_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Query",
            "Response",
            "Ground Truth",
            "Chunking Strategy",
            "Embedding Model",
            "Retrieval Strategy",
            "Reranking Strategy",
            "Vector Store",
            "Status",
            "Error",
            "Process Time (s)",
            "Query Time (s)",
            "Faithfulness",
            "Context Relevancy",
            "Context Recall",
            "Answer Correctness",
            "Answer Relevancy",
            "Overall Score"
        ])
    
    # Counter for progress tracking
    counter = 0
    
    # Test each combination
    for chunker_type, chunker_config in CHUNKER_PARAM_GRID:
        for embedder_type, embedder_config in EMBEDDER_PARAM_GRID:
            for vs_type, vs_config in VECTOR_STORE_PARAM_GRID:
                for retriever_type, retriever_config in RETRIEVER_PARAM_GRID:
                    for reranker_type, reranker_config in RERANKER_PARAM_GRID:
                        for query in TEST_QUERIES:
                            counter += 1
                            print(f"\nEvaluating combination {counter}/{total_combinations}:")
                            print(f"Chunker: {chunker_type}")
                            print(f"Embedder: {embedder_type}")
                            print(f"Vector Store: {vs_type}")
                            print(f"Retriever: {retriever_type}")
                            print(f"Reranker: {reranker_type}")
                            print(f"Query: {query}")
                            
                            # Evaluate the combination
                            result = evaluate_rag_combination(
                                chunker_config,
                                embedder_config,
                                vs_config,
                                retriever_config,
                                reranker_config,
                                query
                            )
                            
                            # Add to results
                            row = [
                                query,
                                result.get("response", ""),
                                GROUND_TRUTHS.get(query, ""),
                                chunker_type,
                                embedder_type,
                                retriever_type,
                                reranker_type,
                                vs_type,
                                result.get("status", ""),
                                result.get("error", ""),
                                round(result.get("process_time", 0), 2),
                                round(result.get("query_time", 0), 2),
                                "",  # Faithfulness (to be filled manually)
                                "",  # Context Relevancy (to be filled manually)
                                "",  # Context Recall (to be filled manually)
                                "",  # Answer Correctness (to be filled manually)
                                "",  # Answer Relevancy (to be filled manually)
                                ""   # Overall Score (to be filled manually)
                            ]
                            
                            # Append to CSV
                            with open(RESULTS_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
                                writer = csv.writer(f)
                                writer.writerow(row)
                            
                            print(f"Result: {result.get('status', 'unknown')}")
                            if result.get("status") == "failed":
                                print(f"Error: {result.get('error', 'unknown error')}")
    
    print(f"\nEvaluation complete. Results saved to {RESULTS_CSV_PATH}")
    
    # Load and return the results DataFrame
    return pd.read_csv(RESULTS_CSV_PATH)

def run_single_combination(
    chunker_type, 
    embedder_type, 
    vector_store_type, 
    retriever_type, 
    reranker_type, 
    query
):
    """
    Run a single combination for testing purposes
    """
    # Find the configurations
    chunker_config = next((c for t, c in CHUNKER_PARAM_GRID if t == chunker_type), None)
    embedder_config = next((c for t, c in EMBEDDER_PARAM_GRID if t == embedder_type), None)
    vs_config = next((c for t, c in VECTOR_STORE_PARAM_GRID if t == vector_store_type), None)
    retriever_config = next((c for t, c in RETRIEVER_PARAM_GRID if t == retriever_type), None)
    reranker_config = next((c for t, c in RERANKER_PARAM_GRID if t == reranker_type), None)
    
    if not all([chunker_config, embedder_config, vs_config, retriever_config, reranker_config]):
        print("One or more configurations not found")
        return
    
    # Evaluate the combination
    result = evaluate_rag_combination(
        chunker_config,
        embedder_config,
        vs_config,
        retriever_config,
        reranker_config,
        query
    )
    
    print("\nResults:")
    for key, value in result.items():
        if key != "contexts":  # Skip printing the full contexts
            print(f"{key}: {value}")
    
    return result

if __name__ == "__main__":
    # Run the full evaluation
    run_evaluation()
    
    # Alternatively, test a single combination
    # run_single_combination(
    #     chunker_type=constants.ChunkerType.RECURSIVE.value,
    #     embedder_type=constants.EmbedderType.GEMINI.value,
    #     vector_store_type=constants.VectorStore.FAISS.value,
    #     retriever_type=constants.RetrieverType.SIMILARITY.value,
    #     reranker_type=constants.RerankerType.COSINE.value,
    #     query=TEST_QUERIES[0]
    # )
