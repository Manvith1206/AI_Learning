import time
import logging
from typing import List, Dict, Any, Optional, Tuple

from deepeval import evaluate
from deepeval.metrics import (FaithfulnessMetric, AnswerRelevancyMetric, 
                            ContextualPrecisionMetric, ContextualRecallMetric)
from deepeval.test_case import LLMTestCase
from deepeval.models.llms import GeminiModel

from .base_evaluator import BaseEvaluator
from ..common.component_registry import EVALUATORS_REGISTRY
from ..common import rag_constants as constants

logger = logging.getLogger(__name__)

METRIC_MAP = {
    constants.DeepEvalMetricsConstants.DEEP_EVAL_ANSWER_RELEVANCY: constants.EvaluationMetrics.ANSWER_RELEVANCY.value,
    constants.DeepEvalMetricsConstants.DEEP_EVAL_FAITHFULNESS: constants.EvaluationMetrics.FAITHFULNESS.value,
    constants.DeepEvalMetricsConstants.DEEP_EVAL_CONTEXT_PRECISION: constants.EvaluationMetrics.CONTEXT_PRECISION.value,
    constants.DeepEvalMetricsConstants.DEEP_EVAL_CONTEXT_RECALL: constants.EvaluationMetrics.CONTEXT_RECALL.value,
}

@EVALUATORS_REGISTRY.register(constants.EvaluatorType.DEEP_EVAL.value)
class DeepEvalEvaluator(BaseEvaluator):
    """Evaluator that uses the DeepEval framework for evaluation."""

    def __init__(self, gemini_api_key: str, metrics: Optional[List[Any]] = None):
        if not gemini_api_key:
            raise ValueError("Gemini API key is required for DeepEvalEvaluator.")
        
        model = GeminiModel(api_key=gemini_api_key)
        self.metrics = metrics or [
            AnswerRelevancyMetric(threshold=0.7, model=model, include_reason=True),
            FaithfulnessMetric(threshold=0.7, model=model, include_reason=True),
            ContextualPrecisionMetric(threshold=0.7, model=model, include_reason=True),
            ContextualRecallMetric(threshold=0.7, model=model, include_reason=True),
        ]
        self._time_taken = 0.0
        self._cost = 0.0

    def evaluate(
        self, 
        question: str, 
        answer: str, 
        contexts: List[str], 
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        self._time_taken = 0.0
        self._cost = 0.0
        start_time = time.time()

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=contexts,
            expected_output=ground_truths[0] if ground_truths else None
        )

        try:
            result = evaluate(test_cases=[test_case], metrics=self.metrics, print_results=False)
            metrics_dict = {}
            total_cost = 0

            for test_result in result.test_results:
                for metric_data in test_result.metrics_data:
                    metric_name = METRIC_MAP.get(metric_data.name, metric_data.name)
                    metrics_dict[metric_name] = round(metric_data.score, 2)
                    total_cost += metric_data.evaluation_cost
            
            self._cost = total_cost
        except Exception as e:
            logger.error(f"DeepEval evaluation failed: {e}", exc_info=True)
            metrics_dict = {metric.name: 0.0 for metric in self.metrics}

        self._time_taken = time.time() - start_time
        return metrics_dict

    def get_cost_and_time_taken(self) -> Tuple[float, float]:
        return self._cost, self._time_taken
    