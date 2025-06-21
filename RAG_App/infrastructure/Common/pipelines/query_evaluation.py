import infrastructure.common.rag_constants as constants
import Utils.exceptions as Exceptions
from infrastructure.evaluators.base_evaluator import BaseEvaluator

class QueryEvaluation:
    def __init__(self, evaluator: BaseEvaluator):
        self.evaluator = evaluator

    def evaluate(self, question=None, answer=None, contexts=None, ground_truths=None):
        """Evaluate the RAG system using the configured evaluator
        
        Args:
            question: The question to evaluate (uses last query if None)
            answer: The answer to evaluate (uses last query if None)
            contexts: The contexts to evaluate (uses last query if None)
            ground_truths: Optional ground truth answers
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Use last query data if not provided
            
            # if hasattr(self, constants.LAST_QUERY) and (question is None or answer is None or contexts is None):
            #     question = question or self.last_query[constants.QUESTION]
            #     answer = answer or self.last_query[constants.ANSWER]
            #     contexts = contexts or self.last_query[constants.CONTEXTS]
            
            if not (question and answer and contexts):
                raise ValueError("No query data available for evaluation")
            
            # Run evaluation
            metrics = self.evaluator.evaluate(question, answer, contexts, ground_truths)
            return metrics
        except Exception as e:
            raise Exceptions.EvaluationError("Error During Evaluation")
