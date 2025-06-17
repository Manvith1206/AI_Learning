import infrastructure.Common.RAG_Constants as constants
from infrastructure.Common.RAG_Constants import (
    ChunkerType, EmbedderType,
    RetrieverType, RerankerType,
    EvaluatorType, LLMServiceType
)
from UI.pages.main_page import MainPage

configs = [
  {
    constants.CONFIG_CHUNKER: {
      constants.CONFIG_TYPE_PARAM: ChunkerType.RECURSIVE.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_CHUNK_SIZE_PARAM: 150,
        constants.CONFIG_CHUNK_OVERLAP_PARAM: 70
      }
    },
    constants.CONFIG_EMBEDDER: {
      constants.CONFIG_TYPE_PARAM: EmbedderType.VOYAGE.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_MODEL: constants.VoyageEmbedModels.VOYAGE_3_LITE_EMBED_MODEL.value
      }
    },
    constants.CONFIG_VECTOR_STORE: {
      constants.CONFIG_TYPE_PARAM: constants.CONFIG_VECTOR_STORE_FAISS,
      constants.CONFIG_PARAM: {
          
      }
    },
    constants.CONFIG_RETRIEVER: {
      constants.CONFIG_TYPE_PARAM: RetrieverType.SIMILARITY.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_SIMILARITY_THRESHOLD_PARAM: 0.0,
        constants.CONFIG_TOP_K_PARAM: 5
      }
    },
    constants.CONFIG_RERANKER: {
      constants.CONFIG_TYPE_PARAM: RerankerType.COHERE.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_TOP_K_FOR_RERANKING_PARAM: 5,
        constants.CONFIG_MODEL: constants.CohereLLMModel.RERANK_DEFAULT_MODEL.value
      }
    },
    constants.CONFIG_LLM: {
      constants.CONFIG_TYPE_PARAM: LLMServiceType.GEMINI.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_MODEL: constants.GeminiLLMModel.GEMINI_FLASH.value
      }
    },
    constants.CONFIG_EVALUATOR: {
      constants.CONFIG_TYPE_PARAM: EvaluatorType.RAGAS.value
    }
  },
  {
    constants.CONFIG_CHUNKER: {
      constants.CONFIG_TYPE_PARAM: ChunkerType.RECURSIVE.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_CHUNK_SIZE_PARAM: 150,
        constants.CONFIG_CHUNK_OVERLAP_PARAM: 70
      }
    },
    constants.CONFIG_EMBEDDER: {
      constants.CONFIG_TYPE_PARAM: EmbedderType.VOYAGE.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_MODEL: constants.VoyageEmbedModels.VOYAGE_3_EMBED_MODEL.value
      }
    },
    constants.CONFIG_VECTOR_STORE: {
      constants.CONFIG_TYPE_PARAM: constants.CONFIG_VECTOR_STORE_FAISS,
      constants.CONFIG_PARAM: {
          
      }
    },
    constants.CONFIG_RETRIEVER: {
      constants.CONFIG_TYPE_PARAM: RetrieverType.SIMILARITY.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_SIMILARITY_THRESHOLD_PARAM: 0.0,
        constants.CONFIG_TOP_K_PARAM: 5
      }
    },
    constants.CONFIG_RERANKER: {
      constants.CONFIG_TYPE_PARAM: RerankerType.LLM.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_TOP_K_FOR_RERANKING_PARAM: 5
      }
    },
    constants.CONFIG_LLM: {
      constants.CONFIG_TYPE_PARAM: LLMServiceType.CLAUDE.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_MODEL: constants.CLAUDE_MODELS.CLAUDE_SONNET_THREE_7.value
      }
    },
    constants.CONFIG_EVALUATOR: {
      constants.CONFIG_TYPE_PARAM: EvaluatorType.RAGAS.value
    }
  },
  {
    constants.CONFIG_CHUNKER: {
      constants.CONFIG_TYPE_PARAM: ChunkerType.RECURSIVE.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_CHUNK_SIZE_PARAM: 150,
        constants.CONFIG_CHUNK_OVERLAP_PARAM: 70
      }
    },
    constants.CONFIG_EMBEDDER: {
      constants.CONFIG_TYPE_PARAM: EmbedderType.VOYAGE.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_MODEL: constants.VoyageEmbedModels.VOYAGE_EMBED_DEFAULT_MODEL.value
      }
    },
    constants.CONFIG_VECTOR_STORE: {
      constants.CONFIG_TYPE_PARAM: constants.CONFIG_VECTOR_STORE_FAISS,
      constants.CONFIG_PARAM: {
          
      }
    },
    constants.CONFIG_RETRIEVER: {
      constants.CONFIG_TYPE_PARAM: RetrieverType.SIMILARITY.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_SIMILARITY_THRESHOLD_PARAM: 0.0,
        constants.CONFIG_TOP_K_PARAM: 5
      }
    },
    constants.CONFIG_RERANKER: {
      constants.CONFIG_TYPE_PARAM: RerankerType.LLM.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_TOP_K_FOR_RERANKING_PARAM: 5
      }
    },
    constants.CONFIG_LLM: {
      constants.CONFIG_TYPE_PARAM: LLMServiceType.CLAUDE.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_MODEL: constants.CLAUDE_MODELS.CLAUDE_SONNET_THREE_7.value
      }
    },
    constants.CONFIG_EVALUATOR: {
      constants.CONFIG_TYPE_PARAM: EvaluatorType.RAGAS.value
    }
  },
  {
    constants.CONFIG_CHUNKER: {
      constants.CONFIG_TYPE_PARAM: ChunkerType.SENTENCE.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_MAX_SENTENCES: 18
      }
    },
    constants.CONFIG_EMBEDDER: {
      constants.CONFIG_TYPE_PARAM: EmbedderType.COHERE.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_MODEL: constants.CohereEmbedModels.COHERE_EMBED_MODEL_DEFAULT.value
      }
    },
    constants.CONFIG_VECTOR_STORE: {
      constants.CONFIG_TYPE_PARAM: constants.CONFIG_VECTOR_STORE_FAISS,
      constants.CONFIG_PARAM: {}
    },
    constants.CONFIG_RETRIEVER: {
      constants.CONFIG_TYPE_PARAM: RetrieverType.SIMILARITY.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_SIMILARITY_THRESHOLD_PARAM: 0.0,
        constants.CONFIG_TOP_K_PARAM: 5
      }
    },
    constants.CONFIG_RERANKER: {
      constants.CONFIG_TYPE_PARAM: RerankerType.LLM.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_TOP_K_FOR_RERANKING_PARAM: 5
      }
    },
    constants.CONFIG_LLM: {
      constants.CONFIG_TYPE_PARAM: LLMServiceType.CLAUDE.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_MODEL: constants.CLAUDE_MODELS.CLAUDE_SONNET_THREE_7.value
      }
    },
    constants.CONFIG_EVALUATOR: {
      constants.CONFIG_TYPE_PARAM: EvaluatorType.RAGAS.value
    }
  },
  {
    constants.CONFIG_CHUNKER: {
      constants.CONFIG_TYPE_PARAM: ChunkerType.SENTENCE.value,
      constants.CONFIG_PARAM: {
          
        constants.CONFIG_MAX_SENTENCES: 18
      }
    },
    constants.CONFIG_EMBEDDER: {
      constants.CONFIG_TYPE_PARAM: EmbedderType.COHERE.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_MODEL: constants.CohereEmbedModels.COHERE_EMBED_MODEL_ENG.value
      }
    },
    constants.CONFIG_VECTOR_STORE: {
      constants.CONFIG_TYPE_PARAM: constants.CONFIG_VECTOR_STORE_FAISS,
      constants.CONFIG_PARAM: {
          
      }
    },
    constants.CONFIG_RETRIEVER: {
      constants.CONFIG_TYPE_PARAM: RetrieverType.HYBRID.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_KEYWORD_WEIGHT: 0.45,
        constants.CONFIG_TOP_K_PARAM: 5
      }
    },
    constants.CONFIG_RERANKER: {
      constants.CONFIG_TYPE_PARAM: RerankerType.JINA.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_TOP_K_FOR_RERANKING_PARAM: 5,
        constants.CONFIG_MODEL: constants.JINA_RERANKER_MODELS.JINA_RERANKER_MULTILINGUAL.value
      }
    },
    constants.CONFIG_LLM: {
      constants.CONFIG_TYPE_PARAM: LLMServiceType.CLAUDE.value,
      constants.CONFIG_PARAM: {
        constants.CONFIG_MODEL: constants.CLAUDE_MODELS.CLAUDE_SONNET_THREE_7.value
      }
    },
    constants.CONFIG_EVALUATOR: {
      constants.CONFIG_TYPE_PARAM: EvaluatorType.RAGAS.value
    }
  },
]

class DummyFile:
    """Wrapper to mimic Streamlit UploadedFile interface for local tests"""
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)

    def getbuffer(self):
        with open(self.path, 'rb') as f:
            return f.read()
        
    def getvalue(self):
        """Return file contents as bytes (mimics Streamlit's getvalue method)"""
        with open(self.path, 'rb') as f:
            return f.read()
        
import json
TEST_EVAL_SET_PATH = r"C:\Users\ibc-dev\Manvith\AI\RAG_App\infrastructure\Testing\EvalSet.json"
TEST_FILE_PATH = r"C:\Users\ibc-dev\Manvith\AI\RAG_App\infrastructure\Testing\TestFile.pdf"
RESULTS_CSV_PATH = r"C:\Users\ibc-dev\Manvith\AI\RAG_App\infrastructure\Testing\rag_evaluation_results.csv"
import csv
import os
import Utils.Utils as Utils
from config import ConfigManager

def test_rag_combinations():
    
    ragPipeline = Utils.get_pipeline()
    config_manager = ConfigManager()
    mainPage = MainPage(ragPipeline, config_manager=config_manager)

        
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
            "Answer Relevancy",
        ])
    # Open and load the JSON file
    with open(TEST_EVAL_SET_PATH, "r") as file:
        data = json.load(file)
    # Accessing the content
    for item in data:
        for config in configs:
            ragPipeline.update_component(constants.CONFIG_CHUNKER, config[constants.CONFIG_CHUNKER])
            ragPipeline.update_component(constants.CONFIG_EMBEDDER, config[constants.CONFIG_EMBEDDER])
            ragPipeline.update_component(constants.CONFIG_VECTOR_STORE, config[constants.CONFIG_VECTOR_STORE])
            ragPipeline.update_component(constants.CONFIG_RETRIEVER, config[constants.CONFIG_RETRIEVER])
            ragPipeline.update_component(constants.CONFIG_RERANKER, config[constants.CONFIG_RERANKER])
            ragPipeline.update_component(constants.CONFIG_LLM, config[constants.CONFIG_LLM])
            ragPipeline.update_component(constants.CONFIG_EVALUATOR, config[constants.CONFIG_EVALUATOR])
            mainPage.load_pre_processed_docs_or_process_the_doc(DummyFile(TEST_FILE_PATH))
            
            response = ragPipeline.query(item["Question"], "")
            contexts = None
            full_response = ""
            for delta in response:
                print("Delta: ", delta)
                full_response += delta[constants.ANSWER]
                contexts = delta[constants.CONTEXTS]
            breakpoint()
            metric_iter_dict = {}
            metrics_dict = {}
            for i in range(0, 3):
                breakpoint()
                result = ragPipeline.evaluate(
                    question=item["Question"], 
                    answer=full_response, 
                    contexts=contexts, 
                    ground_truths=item["Ground_Truth"]
                )
                metric_iter_dict[f"iteration_{i}"] = result
            faithfulness = [
            metrics[constants.FAITHFULNESS] for metrics in metric_iter_dict.values()
            ]
            context_recall = [
            metrics[constants.CONTEXT_RECALL] for metrics in metric_iter_dict.values()
            ]
            context_precision = [
            metrics[constants.CONTEXT_PRECISION] for metrics in metric_iter_dict.values()
            ]
            answer_relavancy = [
            metrics[constants.ANSWER_RELEVANCY] for metrics in metric_iter_dict.values()
            ]
            
            chunker_cost, chunker_time = ragPipeline.get_chunker_cost_and_time()
            embedder_cost, embedder_time = ragPipeline.get_embedder_cost_and_time()
            vector_store_cost, vector_store_time = ragPipeline.get_vector_store_cost_and_time()
            reranker_cost, reranker_time = ragPipeline.get_reranker_cost_and_time()
            retriever_cost, retriever_time = ragPipeline.get_retriever_cost_and_time()
            llm_service_cost, llm_service_time = ragPipeline.get_llm_service_cost_and_time()
           

            faithfulness = sum(faithfulness) / len(faithfulness)
            context_recall = sum(context_recall) / len(context_recall)
            context_precision = sum(context_precision) / len(context_precision)
            answer_relavancy = sum(answer_relavancy) / len(answer_relavancy)

            chunker_name = f"{config[constants.CONFIG_CHUNKER][constants.CONFIG_TYPE_PARAM]}_{config[constants.CONFIG_CHUNKER][constants.CONFIG_PARAM]}\nCost: {chunker_cost}"
            retriever_name = f"{config[constants.CONFIG_RETRIEVER][constants.CONFIG_TYPE_PARAM]}_{config[constants.CONFIG_RETRIEVER][constants.CONFIG_PARAM]}\nCost: {retriever_cost}"
            reranker_name = f"{config[constants.CONFIG_RERANKER][constants.CONFIG_TYPE_PARAM]}_{config[constants.CONFIG_RERANKER][constants.CONFIG_PARAM]}\nCost: {reranker_cost}"
            vs_name = f"{config[constants.CONFIG_VECTOR_STORE][constants.CONFIG_TYPE_PARAM]}_{config[constants.CONFIG_VECTOR_STORE][constants.CONFIG_PARAM]}\nCost: {vector_store_cost}"
            llm_service_name = f"{config[constants.CONFIG_LLM][constants.CONFIG_TYPE_PARAM]}_{config[constants.CONFIG_LLM][constants.CONFIG_PARAM]}\nCost: {llm_service_cost}"
            embedder_name = f"{config[constants.CONFIG_EMBEDDER][constants.CONFIG_TYPE_PARAM]}_{config[constants.CONFIG_EMBEDDER][constants.CONFIG_PARAM]}\nCost: {embedder_cost}"
            
            with open(RESULTS_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                                item["Question"],
                                response[constants.ANSWER],
                                item["Ground_Truth"],  # Ground Truth
                                chunker_name,
                                embedder_name,
                                retriever_name,
                                reranker_name,
                                vs_name,
                                llm_service_name,
                                faithfulness,
                                context_precision,
                                context_recall,
                                answer_relavancy
                            ])

