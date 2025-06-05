import time


from .base_evaluator import BaseEvaluator
from ragas.metrics import answer_relevancy, faithfulness, answer_correctness, context_precision, context_recall
from ragas import evaluate
from datasets import Dataset
import infrastructure.Common.RAG_Constants as constants
from ragas.dataset_schema import MultiTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from ragas.llms import LangchainLLMWrapper

class RagasEvaluator(BaseEvaluator):
    """Evaluator that uses RAGAS metrics for RAG evaluation"""
    
    def __init__(self, metrics=None, gemini_api_key: str = None, openai_api_key: str = None):
        """
        Initialize with specific metrics or use default
        
        Args:
            metrics: List of RAGAS metrics to use (default: faithfulness)
        """
        if not gemini_api_key:
            raise ValueError("Gemini API key must be provided for RagasEvaluator if Gemini models are used.")
        if not openai_api_key:
            raise ValueError("OpenAI API key must be provided for RagasEvaluator if OpenAI models are used.")
        self.gemini_api_key = gemini_api_key
        self.openai_api_key = openai_api_key
        self.metrics = metrics or [faithfulness, context_precision, answer_correctness, context_recall, answer_relevancy]
        self.time_taken = 0
        self.cost = 0

    def evaluate(self, question, answer, contexts, ground_truths=None):
        """
        Evaluate using RAGAS metrics
        
        Args:
            question: The query/question asked
            answer: The generated answer
            contexts: The contexts used to generate the answer
            ground_truths: Optional ground truth answers (list)
            
        Returns:
            Dictionary of evaluation metrics
        """
        start_time = time.time()  
        questions = [question]
        answers = [answer]
        contexts_list = [contexts]
        ground_truths_list = [ground_truths]


        data = Dataset.from_dict({
            constants.QUESTION: questions,
            constants.ANSWER: answers,
            constants.CONTEXTS: contexts_list,
            "ground_truth": ground_truths_list
        })
        print("Question: ", question)
        print("Answer: ", answer)
        print("Contexts: ", contexts_list)
        print("Ground Truths: ", ground_truths)

        chatLLM = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
            api_key=self.openai_api_key
        )
        geminiLLM = ChatGoogleGenerativeAI(
            model=constants.GeminiLLMModel.GEMINI_FLASH.value,
            temperature=0.0,
            google_api_key=self.gemini_api_key
        )
        print("Using LLM: ", geminiLLM)
        print("Using OpenAI LLM: ", chatLLM)
        result = evaluate(
                data,
                metrics=self.metrics,
                raise_exceptions=True,
                llm = chatLLM
            )
            
        print("Cost", result.cost_cb)
        print("TotalCost", result.total_cost)

        metrics_dict = {}
        metrics_dict[constants.FAITHFULNESS] = round((result[constants.FAITHFULNESS][0]), 2)
        metrics_dict[constants.CONTEXT_PRECISION] = round((result[constants.CONTEXT_PRECISION][0]), 2)
        metrics_dict[constants.CONTEXT_RECALL] = round((result[constants.CONTEXT_RECALL][0]), 2)
        metrics_dict[constants.ANSWER_RELEVANCY] = round((result[constants.ANSWER_RELEVANCY][0]), 2)
        end_time = time.time()
        self.time_taken = end_time - start_time
        return metrics_dict
    
    def get_cost_and_time_taken(self):
        """
        Get the cost and time taken for the evaluation
        """
        return self.cost, self.time_taken