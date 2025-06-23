import infrastructure.common.rag_constants as constants
from infrastructure.common.rag_constants import (
    ChunkerType, EmbedderType,
    RetrieverType, RerankerType,
    EvaluatorType, LLMServiceType
)
from UI.pages.main_page import MainPage
from infrastructure.testing.config_generator import generate_configurations
generate_configurations()
configs = []

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
CONFIGS_PATH = r"C:\Users\ibc-dev\Manvith\AI\RAG_App\infrastructure\Testing\Configs.json"
TEST_EVAL_SET_PATH = r"C:\Users\ibc-dev\Manvith\AI\RAG_App\infrastructure\Testing\EvalSet.json"
TEST_FILE_PATH = r"C:\Users\ibc-dev\Manvith\AI\RAG_App\infrastructure\Testing\TestFile.pdf"
RESULTS_CSV_PATH = r"C:\Users\ibc-dev\Manvith\AI\RAG_App\infrastructure\Testing\rag_evaluation_results.csv"
import csv
import os
import Utils.utils as Utils
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
            "Contexts",
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
    with open(CONFIGS_PATH, "r") as file:
        configs = json.load(file)

    # Accessing the content
    for item in data:
        for config in configs:
            ragPipeline.component_manager.update_component(constants.CONFIG_CHUNKER, config[constants.CONFIG_CHUNKER])
            ragPipeline.component_manager.update_component(constants.CONFIG_EMBEDDER, config[constants.CONFIG_EMBEDDER])
            ragPipeline.component_manager.update_component(constants.CONFIG_VECTOR_STORE, config[constants.CONFIG_VECTOR_STORE])
            ragPipeline.component_manager.update_component(constants.CONFIG_RETRIEVER, config[constants.CONFIG_RETRIEVER])
            ragPipeline.component_manager.update_component(constants.CONFIG_RERANKER, config[constants.CONFIG_RERANKER])
            ragPipeline.component_manager.update_component(constants.CONFIG_LLM, config[constants.CONFIG_LLM])
            ragPipeline.component_manager.update_component(constants.CONFIG_EVALUATOR, config[constants.CONFIG_EVALUATOR])
            mainPage.load_pre_processed_docs_or_process_the_doc(DummyFile(TEST_FILE_PATH))
            
            response = ragPipeline.query(item["Question"], "")
            contexts = None
            full_response = ""
            for delta in response:
                print("Delta: ", delta)
                full_response = delta[constants.ANSWER]
                contexts = delta[constants.CONTEXTS]
            
            metric_iter_dict = {}
            metrics_dict = {}
            # for i in range(0, 3):
                
            #     result = ragPipeline.evaluate(
            #         question=item["Question"], 
            #         answer=full_response, 
            #         contexts=[contexts], 
            #         ground_truths=item["Ground_Truth"]
            #     )
            #     metric_iter_dict[f"iteration_{i}"] = result
            # faithfulness = [
            # metrics[constants.FAITHFULNESS] for metrics in metric_iter_dict.values()
            # ]
            # context_recall = [
            # metrics[constants.CONTEXT_RECALL] for metrics in metric_iter_dict.values()
            # ]
            # context_precision = [
            # metrics[constants.CONTEXT_PRECISION] for metrics in metric_iter_dict.values()
            # ]
            # answer_relavancy = [
            # metrics[constants.ANSWER_RELEVANCY] for metrics in metric_iter_dict.values()
            # ]
            result = ragPipeline.evaluate(
                    question=item["Question"], 
                    answer=full_response, 
                    contexts=[contexts], 
                    ground_truths=item["Ground_Truth"]
                )
            chunker_cost, chunker_time = ragPipeline.component_manager.get_chunker_cost_and_time()
            embedder_cost, embedder_time = ragPipeline.component_manager.get_embedder_cost_and_time()
            vector_store_cost, vector_store_time = ragPipeline.component_manager.get_vector_store_cost_and_time()
            reranker_cost, reranker_time = ragPipeline.component_manager.get_reranker_cost_and_time()
            retriever_cost, retriever_time = ragPipeline.component_manager.get_retriever_cost_and_time()
            llm_service_cost, llm_service_time = ragPipeline.component_manager.get_llm_service_cost_and_time()
           

            faithfulness = result[constants.FAITHFULNESS]
            context_recall = result[constants.CONTEXT_RECALL]
            context_precision = result[constants.CONTEXT_PRECISION]
            answer_relavancy = result[constants.ANSWER_RELEVANCY]

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
                                full_response,
                                item["Ground_Truth"],  # Ground Truth,
                                contexts,
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

