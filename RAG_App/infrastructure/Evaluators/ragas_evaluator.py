import time
from .base_evaluator import BaseEvaluator
from ragas.metrics import answer_relevancy, faithfulness, answer_correctness, context_precision, context_recall
from ragas import evaluate
from datasets import Dataset
import os
import openai
import infrastructure.Common.RAG_Constants as constants
from ragas.dataset_schema import MultiTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from UI.UI_Components import UIComponents

from ragas.llms import LangchainLLMWrapper

class RagasEvaluator(BaseEvaluator):
    """Evaluator that uses RAGAS metrics for RAG evaluation"""
    
    def __init__(self, metrics=None):
        """
        Initialize with specific metrics or use default
        
        Args:
            metrics: List of RAGAS metrics to use (default: faithfulness)
        """
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

        chatLLM = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
        )
        
        with UIComponents.display_spinner("Running RAG evaluation..."):
            result = evaluate(
                data,
                metrics=self.metrics,
                raise_exceptions=True,
                llm = chatLLM
            )

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