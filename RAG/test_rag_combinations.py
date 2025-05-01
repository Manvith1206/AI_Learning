import os
import sys
import csv
import time
import pandas as pd
import streamlit as st

# Add project root to sys.path
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from rag_modular.rag_pipeline import RAGPipeline
from rag_modular.config_manager import ConfigManager
import rag_modular.RAG_Constants as constants
from rag_modular.RAG_Constants import (
    ChunkerType, EmbedderType, RetrieverType, 
    RerankerType, VectorStore, GeminiLLMModel,
    CohereLLMModel, JINA_RERANKER_MODELS
)
from rag_modular.ragas_evaluator import RagasEvaluator

# Path to the test document and output file
TEST_FILE_PATH = os.path.join(ROOT, "rag_modular", "TestFile", "DCA2104 Unit-08_V1.1.txt")
RESULTS_CSV_PATH = os.path.join(ROOT, "rag_evaluation_results.csv")

if "Extractedtexts" not in st.session_state:
    st.session_state.Extractedtexts = None

# Test query to use for all combinations
TEST_QUERY = "What is Synchronous Transmission?"
GROUND_TRUTH = "By using synchronous transmission, we can transmit large block of bits in a steady stream without start and stop codes. The block may be many bits in length. To prevent timing change between transmitter and receiver, their clocks must be synchronized... In case of synchronous transmission, there is another level of synchronization required to allow the receiver to determine the beginning and end of a block of data. To achieve this, each block begins with a preamble bit pattern and ends with a postamble bit pattern. Also, other bits are added to the block that convey control information used in the data link control procedures. For sizable blocks of data, synchronous transmission is far more efficient than asynchronous... The data plus preamble, postamble, and control information are called a frame."

class DummyFile:
    """Wrapper to mimic Streamlit UploadedFile interface for local tests"""
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)

    def getbuffer(self):
        with open(self.path, 'rb') as f:
            return f.read()

def test_rag_combination(config_manager, chunker_config, embedder_config, 
                         vector_store_config, retriever_config, reranker_config, llm_service_config, query, pipeline, texts):
    """Test a specific RAG combination with the given configurations"""
    try:
        # Initialize pipeline with the specific configuration
        
        

        # Process document
        print(f"Processing document...")
        docs, chunks = pipeline.process_document(DummyFile(TEST_FILE_PATH), texts)
        
        if not docs or not chunks:
            return {
                constants.STATUS_CONFIG_NAME: "failed",
                constants.ERROR_CONFIG_NAME: "No documents or chunks produced",
                constants.RESPONSE_CONFIG_NAME: "",
                constants.CONTEXTS_CONFIG_NAME: "",
                constants.METRICS_CONFIG_NAME: {}
            }
        
        # Run query
        print(f"Running query: {query}")
        response = pipeline.query(query)
        
        # Run RAGAS evaluation
        print(f"Running RAGAS evaluation...")
        metrics = {}
        try:
            # Create RAGAS evaluator
            ragas_evaluator = RagasEvaluator()
            
            # Get the response and contexts
            answer = response[constants.ANSWER]
            contexts = response[constants.CONTEXTS]
            # Run evaluation
            metrics = ragas_evaluator.evaluate(
                question=query,
                answer=answer,
                contexts=[contexts] if isinstance(contexts, str) else contexts,
                ground_truths=GROUND_TRUTH
            )
            
            print(f"RAGAS metrics: {metrics}")
        except Exception as eval_error:
            print(f"Error in RAGAS evaluation: {str(eval_error)}")
        
        return {
            constants.STATUS_CONFIG_NAME: "success",
            constants.RESPONSE_CONFIG_NAME: response[constants.ANSWER],
            constants.CONTEXTS_CONFIG_NAME: response[constants.CONTEXTS],
            constants.ERROR_CONFIG_NAME: "",
            constants.GROUND_TRUTH_CONFIG_NAME: GROUND_TRUTH,
            constants.METRICS_CONFIG_NAME: metrics
        }
    
    except Exception as e:
        print(f"Error in combination: {str(e)}")
        return {
            constants.STATUS_CONFIG_NAME: "failed",
            constants.ERROR_CONFIG_NAME: str(e),
            constants.RESPONSE_CONFIG_NAME: "",
            constants.CONTEXTS_CONFIG_NAME: "",
            constants.GROUND_TRUTH_CONFIG_NAME: GROUND_TRUTH,
            constants.METRICS_CONFIG_NAME: {}
        }

def run_tests():
    """Run tests for all combinations and save results to CSV"""
    # Define the combinations to test
    chunker_configs = [
        # (ChunkerType.RECURSIVE.value, {
        #     constants.CONFIG_TYPE_PARAM: ChunkerType.RECURSIVE.value, 
        #     constants.CONFIG_PARAM: {
        #         constants.CONFIG_CHUNK_SIZE_PARAM: 600, 
        #         constants.CONFIG_CHUNK_OVERLAP_PARAM: 200
        #     }
        # }),
        (ChunkerType.SEMANTIC.value, {
            constants.CONFIG_TYPE_PARAM: ChunkerType.SEMANTIC.value, 
            constants.CONFIG_PARAM: {
                constants.CONFIG_MIN_CHUNK_SIZE_DISPLAY_NAME: 150, 
                constants.CONFIG_MAX_CHUNK_SIZE_DISPLAY_NAME: 550, 
                constants.CONFIG_SIMILARITY_THRESHOLD_PARAM: 0.7, 
                constants.CONFIG_MODEL_NAME: constants.SENTENCE_TRANSFORMER_MODEL_ALL_MINI
            }
        }),
        (ChunkerType.SENTENCE.value, {
            constants.CONFIG_TYPE_PARAM: ChunkerType.SENTENCE.value, 
            constants.CONFIG_PARAM: {
                constants.CONFIG_MAX_SENTENCES: 5
            }
        })
    ]
    
    embedder_configs = [
        # (EmbedderType.TFIDF.value, {
        #     constants.CONFIG_TYPE_PARAM: EmbedderType.TFIDF.value
        # }),
        # (EmbedderType.GEMINI.value, {
        #     constants.CONFIG_TYPE_PARAM: EmbedderType.GEMINI.value, 
        #     constants.CONFIG_MODEL: constants.GeminiEmbedModels.GEMINI_TEXT_EMBED_MODEL.value,
        #     constants.CONFIG_BATCH_SIZE: 0
        #  }),
        (EmbedderType.COHERE.value, {
            constants.CONFIG_TYPE_PARAM: EmbedderType.COHERE.value, 
            constants.CONFIG_MODEL: constants.CohereEmbedModels.COHERE_EMBED_MODEL_DEFAULT.value
        }),
        (EmbedderType.VOYAGE.value, {
            constants.CONFIG_TYPE_PARAM: EmbedderType.VOYAGE.value, 
            constants.CONFIG_MODEL: constants.VoyageEmbedModels.VOYAGE_EMBED_DEFAULT_MODEL.value
        }),
        # (EmbedderType.MISTRAL.value, {
        #     constants.CONFIG_TYPE_PARAM: EmbedderType.MISTRAL.value, 
        #     constants.CONFIG_MODEL: constants.MISTRAL_EMBED_MODELS.MISTRAL_EMBED_MODEL_DEFAULT.value
        # })
    ]
    
    vector_store_configs = [
        (VectorStore.FAISS.value, {
            constants.CONFIG_TYPE_PARAM: constants.CONFIG_VECTOR_STORE_FAISS
        }),
        # (VectorStore.SCIKIT_LEARN.value, {
        #     constants.CONFIG_TYPE_PARAM: constants.CONFIG_VECTOR_STORE_SKLEARN,
        #     constants.CONFIG_VECTOR_STORE_METRIC: constants.CONFIG_METRIC_COSINE
        # }),
        (VectorStore.PINE_CONE.value, {
            constants.CONFIG_TYPE_PARAM: constants.CONFIG_VECTOR_STORE_PINCONE
        })
    ]
    
    retriever_configs = [
        # (RetrieverType.SIMILARITY.value, {
        #     constants.CONFIG_TYPE_PARAM: RetrieverType.SIMILARITY.value, 
        #     constants.CONFIG_PARAM: {
        #         constants.CONFIG_SIMILARITY_THRESHOLD_PARAM: 0.0
        #     },
        #     constants.CONFIG_TOP_K_PARAM: 3
        # }),
        (RetrieverType.HYBRID.value, {
            constants.CONFIG_TYPE_PARAM: RetrieverType.HYBRID.value, 
            constants.CONFIG_PARAM: {
                constants.CONFIG_KEYWORD_WEIGHT: 0.45
            },
            constants.CONFIG_TOP_K_PARAM: 3
        }),
        (RetrieverType.SENTENCE_WINDOW.value, {
            constants.CONFIG_TYPE_PARAM: RetrieverType.SENTENCE_WINDOW.value, 
            constants.CONFIG_PARAM: {
                constants.CONFIG_WINDOW_SIZE: 3
            },
            constants.CONFIG_TOP_K_PARAM: 3
        }),
        
    ]
    
    reranker_configs = [
         
        # (RerankerType.LLM.value, {
        #     constants.CONFIG_TYPE_PARAM: RerankerType.LLM.value, 
        #     constants.CONFIG_PARAM: GeminiLLMModel.GEMINI_FLASH.value
        # }),
        (RerankerType.JINA.value, {
            constants.CONFIG_TYPE_PARAM: RerankerType.JINA.value, 
            constants.CONFIG_PARAM: JINA_RERANKER_MODELS.JINA_RERANKER_MULTILINGUAL.value
        }),
        
        (RerankerType.COHERE.value, {
            constants.CONFIG_TYPE_PARAM: RerankerType.COHERE.value, 
            constants.CONFIG_PARAM: CohereLLMModel.RERANK_DEFAULT_MODEL.value
        }),
    ]

    llm_service_configs = [(constants.LLMServiceType.CLAUDE.value, {
        constants.CONFIG_TYPE_PARAM: constants.LLMServiceType.CLAUDE.value, 
        constants.CONFIG_MODEL: constants.CLAUDE_MODELS.CLAUDE_SONNET_THREE_7.value
    })]
    
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
            "LLM Chat Service",
            "Faithfulness",
            "Context Relevancy",
            "Context Recall",
            "Answer Correctness",
            "Answer Relevancy",
            "Overall Score"
        ])
    
    # Initialize config manager once
    config_manager = ConfigManager()
    
    # Count total combinations
    total_combinations = (
        len(chunker_configs) * 
        len(embedder_configs) * 
        len(vector_store_configs) * 
        len(retriever_configs) * 
        len(reranker_configs) *
        len(llm_service_configs)
    )
    
    print(f"Testing {total_combinations} combinations...")
    
    # Counter for progress tracking
    counter = 0
    pipeline = RAGPipeline(config_manager)
    texts = pipeline.extractText(DummyFile(TEST_FILE_PATH))
    # Test each combination
    for chunker_name, chunker_config in chunker_configs:
        pipeline.update_component(constants.CONFIG_CHUNKER, chunker_config)
        for embedder_name, embedder_config in embedder_configs:
            pipeline.update_component(constants.CONFIG_EMBEDDER, embedder_config)
            for vs_name, vs_config in vector_store_configs:
                pipeline.update_component(constants.CONFIG_VECTOR_STORE, vs_config)
                for retriever_name, retriever_config in retriever_configs:
                    pipeline.update_component(constants.CONFIG_RETRIEVER, retriever_config)
                    for reranker_name, reranker_config in reranker_configs:
                        pipeline.update_component(constants.CONFIG_RERANKER, reranker_config)
                        for llm_service_name, llm_service_config in llm_service_configs:
                            counter += 1
                            print(f"\nTesting combination {counter}/{total_combinations}:")
                            print(f"Chunker: {chunker_name}")
                            print(f"Embedder: {embedder_name}")
                            print(f"Vector Store: {vs_name}")
                            print(f"Retriever: {retriever_name}")
                            print(f"Reranker: {reranker_name}")
                            
                            # Update components with the specific configurations
                            pipeline.update_component(constants.CONFIG_LLM, llm_service_config)
                            # Test the combination
                            result = test_rag_combination(
                                config_manager,
                                chunker_config,
                                embedder_config,
                                vs_config,
                                retriever_config,
                                reranker_config,
                                llm_service_config,
                                TEST_QUERY,
                                pipeline,
                                texts
                            )
                            
                            # Write result to CSV
                            with open(RESULTS_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
                                writer = csv.writer(f)
                                
                                # Get metrics from RAGAS evaluation
                                metrics = result['metrics']
                                faithfulness = metrics.get(constants.FAITHFULNESS, [0])

                                context_precision = metrics.get(constants.CONTEXT_PRECISION, [0])
                                context_recall = metrics.get(constants.CONTEXT_RECALL, [0])
                                answer_correctness = metrics.get(constants.ANSWER_CORRECTNESS, [0])
                                answer_relevancy = metrics.get(constants.ANSWER_RELEVANCY, [0])
                                    
                                # Calculate overall score if all metrics are available
                                overall_score = ""
                                if all(isinstance(m, (int, float)) for m in [
                                    faithfulness, context_precision, context_recall, 
                                    answer_correctness, answer_relevancy
                                ]):
                                    overall_score = round(
                                        (faithfulness + context_precision + context_recall + 
                                        answer_correctness + answer_relevancy) / 5, 
                                        3
                                    )
                                
                                writer.writerow([
                                    TEST_QUERY,
                                    result[constants.RESPONSE_CONFIG_NAME],
                                    result[constants.GROUND_TRUTH_CONFIG_NAME],  # Ground Truth
                                    chunker_name,
                                    embedder_name,
                                    retriever_name,
                                    reranker_name,
                                    vs_name,
                                    llm_service_name,
                                    faithfulness,
                                    context_precision,
                                    context_recall,
                                    answer_correctness,
                                    answer_relevancy,
                                    overall_score
                                ])
                            
                            print(f"Status: {result['status']}")
                            if result["status"] == "failed":
                                print(f"Error: {result['error']}")
    
    print(f"\nTesting complete. Results saved to {RESULTS_CSV_PATH}")
    

if __name__ == "__main__":
    run_tests()
