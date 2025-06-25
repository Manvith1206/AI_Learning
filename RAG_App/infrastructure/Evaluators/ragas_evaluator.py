import time
import logging
from typing import List, Dict, Any, Optional, Tuple

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
    answer_correctness
)
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import get_openai_callback

from .base_evaluator import BaseEvaluator
from ..common.component_registry import EVALUATORS_REGISTRY
from ..common import rag_constants as constants

logger = logging.getLogger(__name__)

METRICS_MAP = {
    constants.RagasMetricsConstants.ANSWER_RELEVANCY: answer_relevancy,
    constants.RagasMetricsConstants.FAITHFULNESS: faithfulness,
    constants.RagasMetricsConstants.CONTEXT_PRECISION: context_precision,
    constants.RagasMetricsConstants.CONTEXT_RECALL: context_recall,
    constants.RagasMetricsConstants.ANSWER_CORRECTNESS: answer_correctness,
}

@EVALUATORS_REGISTRY.register(constants.EvaluatorType.RAGAS.value)
class RagasEvaluator(BaseEvaluator):
    """Evaluator that uses the Ragas framework for evaluation."""

    def __init__(self, openai_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None, metrics: Optional[List[str]] = None):
        self._initialize_llm(openai_api_key, gemini_api_key)
        
        self.metrics_to_run = []
        metric_names = metrics or METRICS_MAP.keys()
        for name in metric_names:
            metric = METRICS_MAP.get(name)
            if metric:
                self.metrics_to_run.append(metric)
            else:
                logger.warning(f"Unknown Ragas metric '{name}' specified.")

        self._time_taken = 0.0
        self._cost = 0.0

    def _initialize_llm(self, openai_api_key: Optional[str], gemini_api_key: Optional[str]):
        if openai_api_key:
            self.llm = ChatOpenAI(model=constants.OPEN_AI_MODELS.GPT_FOUR_O.value, api_key=openai_api_key, temperature=0.0)
        elif gemini_api_key:
            self.llm = ChatGoogleGenerativeAI(model=constants.GeminiLLMModel.GEMINI_PRO.value, google_api_key=gemini_api_key, temperature=0.0)
        else:
            raise ValueError("Either an OpenAI or Gemini API key is required for RagasEvaluator.")

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

        dataset_dict = {"question": [question], "answer": [answer], "contexts": [contexts]}
        metrics = self.metrics_to_run

        if ground_truths:
            dataset_dict["ground_truth"] = [ground_truths[0]]
        else:
            metrics_needing_gt = {answer_correctness, context_recall}
            metrics = [m for m in self.metrics_to_run if m not in metrics_needing_gt]

        if not metrics:
            logger.warning("No Ragas metrics to run after filtering for ground truth availability.")
            self._time_taken = time.time() - start_time
            return {}

        dataset = Dataset.from_dict(dataset_dict)

        try:
            with get_openai_callback() as cb:
                result = evaluate(dataset, metrics=metrics, llm=self.llm, raise_exceptions=True)
                self._cost = cb.total_cost
            
            metrics_dict = {name: round(score, 2) for name, score in result.items()}
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}", exc_info=True)
            metrics_dict = {m.name: 0.0 for m in metrics}

        self._time_taken = time.time() - start_time
        return metrics_dict

    def get_cost_and_time_taken(self) -> Tuple[float, float]:
        return self._cost, self._time_taken