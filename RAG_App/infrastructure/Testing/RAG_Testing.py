import infrastructure.Common.RAG_Constants as constants
from infrastructure.Common.RAG_Constants import (
    ChunkerType, EmbedderType,
    RetrieverType, RerankerType,
    EvaluatorType, LLMServiceType
)
from UI.pages.main_page import MainPage
from infrastructure.Testing.config_generator import generate_configurations

configs = generate_configurations()

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

    # Clear the results file at the beginning of the test run
    if os.path.exists(RESULTS_CSV_PATH):
        os.remove(RESULTS_CSV_PATH)

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

            # Prepare data for CSV logging
            row_data = {
                "Question": item["Question"],
                "Answer": full_response,
                "Ground_Truth": item["Ground_Truth"],
                
                "Chunker_Type": config[constants.CONFIG_CHUNKER][constants.CONFIG_TYPE_PARAM],
                "Chunker_Params": json.dumps(config[constants.CONFIG_CHUNKER][constants.CONFIG_PARAM]),
                "Chunker_Cost": chunker_cost,
                
                "Embedder_Type": config[constants.CONFIG_EMBEDDER][constants.CONFIG_TYPE_PARAM],
                "Embedder_Model": config[constants.CONFIG_EMBEDDER][constants.CONFIG_PARAM].get(constants.CONFIG_MODEL),
                "Embedder_Cost": embedder_cost,
                
                "Vector_Store_Type": config[constants.CONFIG_VECTOR_STORE][constants.CONFIG_TYPE_PARAM],
                "Vector_Store_Cost": vector_store_cost,

                "Retriever_Type": config[constants.CONFIG_RETRIEVER][constants.CONFIG_TYPE_PARAM],
                "Retriever_Params": json.dumps(config[constants.CONFIG_RETRIEVER][constants.CONFIG_PARAM]),
                "Retriever_Cost": retriever_cost,
                
                "Reranker_Type": config[constants.CONFIG_RERANKER][constants.CONFIG_TYPE_PARAM],
                "Reranker_Model": config[constants.CONFIG_RERANKER][constants.CONFIG_PARAM].get(constants.CONFIG_MODEL),
                "Reranker_Cost": reranker_cost,
                
                "LLM_Type": config[constants.CONFIG_LLM][constants.CONFIG_TYPE_PARAM],
                "LLM_Model": config[constants.CONFIG_LLM][constants.CONFIG_PARAM].get(constants.CONFIG_MODEL),
                "LLM_Cost": llm_service_cost,
                
                "Faithfulness": faithfulness,
                "Context_Precision": context_precision,
                "Context_Recall": context_recall,
                "Answer_Relevancy": answer_relavancy
            }

            # Write to CSV
            file_exists = os.path.isfile(RESULTS_CSV_PATH)
            with open(RESULTS_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=row_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)

