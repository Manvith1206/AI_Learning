import time
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from .base_evaluator import BaseEvaluator
from ..common.component_registry import EVALUATORS_REGISTRY
from ..common import rag_constants as constants

logger = logging.getLogger(__name__)

@EVALUATORS_REGISTRY.register(constants.EvaluatorType.SIMPLE.value)
class SimpleEvaluator(BaseEvaluator):
    """
    A simple, heuristic-based evaluator that operates without external dependencies or LLM calls.
    It calculates metrics for context utilization, answer completeness, and ground truth overlap.
    """

    def __init__(self):
        self._time_taken = 0.0
        self._cost = 0.0  # Cost is always zero for this evaluator

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates the RAG output using simple, regex-based heuristics.

        Args:
            question: The input query.
            answer: The generated answer.
            contexts: The retrieved context documents.
            ground_truths: Optional list of ground truth answers.

        Returns:
            A dictionary of calculated metrics.
        """
        self._time_taken = 0.0
        start_time = time.time()
        metrics = {}

        try:
            # 1. Context Utilization
            context_terms = set(word for ctx in contexts for word in re.findall(r'\b\w{4,}\b', ctx.lower()))
            answer_terms = set(re.findall(r'\b\w{4,}\b', answer.lower()))
            
            if context_terms:
                utilization = len(answer_terms.intersection(context_terms)) / len(context_terms)
                metrics["context_utilization"] = min(1.0, utilization * 1.5)  # Scale score
            else:
                metrics["context_utilization"] = 0.0

            # 2. Answer Completeness
            answer_word_count = len(answer.split())
            question_word_count = len(question.split())
            if question_word_count > 0:
                completeness = min(1.0, answer_word_count / (question_word_count * 2.0))
                metrics["answer_completeness"] = completeness
            else:
                metrics["answer_completeness"] = 0.0

            # 3. Ground Truth Overlap
            if ground_truths and ground_truths[0]:
                gt_terms = set(re.findall(r'\b\w{4,}\b', ground_truths[0].lower()))
                if gt_terms:
                    overlap = len(answer_terms.intersection(gt_terms)) / len(gt_terms)
                    metrics["ground_truth_overlap"] = overlap

            # 4. Overall Score
            if metrics:
                metrics["overall_score"] = sum(metrics.values()) / len(metrics)

        except Exception as e:
            logger.error(f"Simple evaluation failed: {e}", exc_info=True)
            # Return zero for all potential metrics in case of failure
            metrics = {
                "context_utilization": 0.0,
                "answer_completeness": 0.0,
                "ground_truth_overlap": 0.0,
                "overall_score": 0.0,
            }

        self._time_taken = time.time() - start_time
        return {k: round(v, 2) for k, v in metrics.items()}

    def get_cost_and_time_taken(self) -> Tuple[float, float]:
        """
        Returns the cost and time taken for the last evaluation. Cost is always 0.
        """
        return self._cost, self._time_taken