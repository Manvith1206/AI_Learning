import time
import re
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

from .base_evaluator import BaseEvaluator
from ..common.component_registry import EVALUATORS_REGISTRY
from ..common import rag_constants as constants
from .llm_evaluation_service import LLMEvaluationService

logger = logging.getLogger(__name__)

class EvaluationMetric(ABC):
    """Abstract base class for an evaluation metric."""
    metric_name: str = "base_metric"

    def __init__(self, llm_service: LLMEvaluationService):
        self.llm_service = llm_service

    @abstractmethod
    def calculate(self, question: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None) -> float:
        """Calculates the metric score."""
        pass

class FaithfulnessMetric(EvaluationMetric):
    metric_name: str = constants.EvaluationMetrics.FAITHFULNESS.value
    # ... (rest of the class is the same)

class ContextPrecisionMetric(EvaluationMetric):
    metric_name: str = constants.EvaluationMetrics.CONTEXT_PRECISION.value
    # ... (rest of the class is the same)

class ContextRecallMetric(EvaluationMetric):
    metric_name: str = constants.EvaluationMetrics.CONTEXT_RECALL.value
    # ... (rest of the class is the same)

class AnswerRelevancyMetric(EvaluationMetric):
    metric_name: str = constants.EvaluationMetrics.ANSWER_RELEVANCY.value
    # ... (rest of the class is the same)

METRIC_REGISTRY = {
    metric.metric_name: metric
    for metric in [FaithfulnessMetric, ContextPrecisionMetric, ContextRecallMetric, AnswerRelevancyMetric]
}

@EVALUATORS_REGISTRY.register(constants.EvaluatorType.CUSTOM.value)
class CustomEvaluator(BaseEvaluator):
    """Evaluator that uses custom, LLM-assisted metrics."""

    def __init__(self, gemini_api_key: str, metrics: Optional[List[str]] = None):
        if not gemini_api_key:
            raise ValueError("Gemini API key is required for CustomEvaluator.")
        
        self.llm_service = LLMEvaluationService(api_key=gemini_api_key)
        self.selected_metrics = []
        
        if metrics:
            for metric_name in metrics:
                metric_class = METRIC_REGISTRY.get(metric_name)
                if metric_class:
                    self.selected_metrics.append(metric_class(llm_service=self.llm_service))
                else:
                    logger.warning(f"Unknown metric '{metric_name}' specified for CustomEvaluator.")
        else:
            # Default to all metrics if none are specified
            for metric_class in METRIC_REGISTRY.values():
                self.selected_metrics.append(metric_class(llm_service=self.llm_service))

        self._cost = 0.0
        self._time_taken = 0.0

    def evaluate(
        self, 
        question: str, 
        answer: str, 
        contexts: List[str], 
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        self._cost = 0.0
        self._time_taken = 0.0
        start_time = time.time()
        results = {}

        # Use the first ground truth if available
        ground_truth = ground_truths[0] if ground_truths else None

        for metric in self.selected_metrics:
            try:
                score = metric.calculate(question, answer, contexts, ground_truth)
                results[metric.metric_name] = score
            except Exception as e:
                logger.error(f"Error calculating metric {metric.metric_name}: {e}", exc_info=True)
                results[metric.metric_name] = None

        self._time_taken = time.time() - start_time
        # Aggregate cost from the LLM service after all calculations
        self._cost = self.llm_service.get_total_cost()
        self.llm_service.reset_cost() # Reset for the next evaluation run
        
        return results

    def get_cost_and_time_taken(self) -> Tuple[float, float]:
        return self._cost, self._time_taken