from .base_evaluator import BaseEvaluator
from ragas.metrics import faithfulness, answer_correctness, context_precision, context_recall, answer_relevancy
from ragas import evaluate
from datasets import Dataset
import os
import streamlit as st
import openai
os.environ["OPENAI_API_KEY"] = st.secrets["OPEN_AI_API_KEY"]

class RagasEvaluator(BaseEvaluator):
    """Evaluator that uses RAGAS metrics for RAG evaluation"""
    
    def __init__(self, metrics=None):
        """
        Initialize with specific metrics or use default
        
        Args:
            metrics: List of RAGAS metrics to use (default: faithfulness)
        """
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
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "reference": [ground_truths]
        })
        

        with st.spinner("Running RAG evaluation..."):
            result = evaluate(
                data,
                metrics=self.metrics
            )
        

        metrics_dict = {}
        metrics_dict["faithfulness"] = result["faithfulness"]
        metrics_dict["answer_correctness"] = result["answer_correctness"]
        metrics_dict["context_precision"] = result["context_precision"]
        metrics_dict["context_recall"] = result["context_recall"]
        metrics_dict["answer_relevancy"] = result["answer_relevancy"]
            
        return metrics_dict
