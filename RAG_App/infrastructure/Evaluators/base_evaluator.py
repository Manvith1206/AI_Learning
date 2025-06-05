from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    """Base class for RAG evaluation metrics"""
    
    @abstractmethod
    def evaluate(self, question, answer, contexts, ground_truths=None):
        """
        Evaluate the RAG system's performance
        
        Args:
            question: The query/question asked
            answer: The generated answer
            contexts: The contexts used to generate the answer
            ground_truths: Optional ground truth answers
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def get_cost_and_time_taken(self):
        """
        Get the cost and time taken for the evaluation process
        """
        pass