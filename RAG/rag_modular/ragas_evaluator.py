from .base_evaluator import BaseEvaluator
from ragas.metrics import answer_relevancy, faithfulness, answer_correctness, context_precision, context_recall
from ragas import evaluate
from datasets import Dataset
import os
import streamlit as st
import rag_modular.RAG_Constants as constants

class RagasEvaluator(BaseEvaluator):
    """Evaluator that uses RAGAS metrics for RAG evaluation"""
    
    def __init__(self, metrics=None):
        """
        Initialize with specific metrics or use default
        
        Args:
            metrics: List of RAGAS metrics to use (default: faithfulness)
        """
        openai_api_key = st.secrets.get(constants.OPENAI_API_KEY) or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("RAGAS evaluation requires OPENAI_API_KEY in Streamlit secrets or environment.")
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.metrics = metrics or [faithfulness, context_precision, answer_correctness, context_recall, answer_relevancy]

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

        with st.spinner("Running RAG evaluation..."):
            result = evaluate(
                data,
                metrics=self.metrics,
                raise_exceptions=True
            )
        

        metrics_dict = {}
        metrics_dict[constants.FAITHFULNESS] = round((result[constants.FAITHFULNESS][0]), 2)
        metrics_dict[constants.CONTEXT_PRECISION] = round((result[constants.CONTEXT_PRECISION][0]), 2)
        metrics_dict[constants.CONTEXT_RECALL] = round((result[constants.CONTEXT_RECALL][0]), 2)
        metrics_dict[constants.ANSWER_RELEVANCY] = round((result[constants.ANSWER_RELEVANCY][0]), 2)
        
        return metrics_dict
