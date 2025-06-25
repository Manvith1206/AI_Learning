from .base_evaluator import BaseEvaluator
import re
import infrastructure.common.rag_constants as constants
from infrastructure.common.component_registry import register, EVALUATORS_REGISTRY

@register(EVALUATORS_REGISTRY, name=constants.EvaluatorType.SIMPLE.value)
class SimpleEvaluator(BaseEvaluator):
    """Simple evaluator that checks basic metrics without external dependencies"""
    
    def evaluate(self, question, answer, contexts, ground_truths=None):
        """
        Evaluate using simple heuristics
        
        Args:
            question: The query/question asked
            answer: The generated answer
            contexts: The contexts used to generate the answer
            ground_truths: Optional ground truth answers
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # 1. Context utilization: Check if answer contains key terms from contexts
        context_terms = set()
        for context in contexts:
            # Extract significant terms (non-stopwords)
            words = re.findall(r'\b\w{4,}\b', context.lower())
            context_terms.update(words)
        
        answer_terms = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        
        if context_terms:
            context_utilization = len(answer_terms.intersection(context_terms)) / len(context_terms)
            metrics["context_utilization"] = min(1.0, context_utilization * 2)  # Scale up a bit
        else:
            metrics["context_utilization"] = 0.0
        
        # 2. Answer completeness: Length relative to question
        answer_words = len(answer.split())
        question_words = len(question.split())
        
        if question_words > 0:
            # Heuristic: answers should be at least 2x question length but not excessively long
            completeness = min(1.0, answer_words / (question_words * 2))
            metrics["answer_completeness"] = completeness
        else:
            metrics["answer_completeness"] = 0.0
        
        # 3. If ground truth available, do simple term overlap
        if ground_truths and ground_truths[0]:
            gt_terms = set(re.findall(r'\b\w{4,}\b', ground_truths[0].lower()))
            if gt_terms:
                term_overlap = len(answer_terms.intersection(gt_terms)) / len(gt_terms)
                metrics["ground_truth_overlap"] = term_overlap  
        
        # Overall score (simple average)
        metrics["overall_score"] = sum(v for v in metrics.values()) / len(metrics)
        
        return metrics

    def get_cost_and_time_taken(self):
        """
        Get the cost and time taken for the evaluation
        """
        # SimpleEvaluator does not have cost or time metrics
        return 0, 0