import time
from .base_evaluator import BaseEvaluator
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    answer_correctness,
    context_precision,
    context_recall,
)
from ragas import evaluate
from datasets import Dataset
import infrastructure.common.rag_constants as constants
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from infrastructure.common.component_registry import register, EVALUATORS_REGISTRY


@register(EVALUATORS_REGISTRY, name=constants.EvaluatorType.RAGAS.value)
class RagasEvaluator(BaseEvaluator):
    """Evaluator that uses RAGAS metrics for RAG evaluation"""

    def __init__(self, openai_api_key: str = None, gemini_api_key: str = None, metrics=None):
        self.metrics = metrics or [
            faithfulness,
            context_precision,
            answer_correctness,
            context_recall,
            answer_relevancy,
        ]

        if openai_api_key:
            self.llm = ChatOpenAI(
                model=constants.OPEN_AI_MODELS.GPT_FOUR_1.value,
                temperature=0.0,
                api_key=openai_api_key,
            )
        elif gemini_api_key:
            self.llm = ChatGoogleGenerativeAI(
                model=constants.GeminiLLMModel.GEMINI_PRO.value,
                temperature=0.0,
                google_api_key=gemini_api_key,
            )
        else:
            raise ValueError("Either OpenAI or Gemini API key is required for RagasEvaluator.")

        self.time_taken = 0
        self.cost = 0

    def evaluate(self, question, answer, contexts, ground_truths=None):
        start_time = time.time()

        metrics_to_run = self.metrics
        dataset_dict = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }

        if ground_truths:
            dataset_dict["ground_truth"] = [ground_truths]
        else:
            # Filter out metrics that require ground_truth
            metrics_that_need_ground_truth = [
                constants.RagasMetricsConstants.ANSWER_CORRECTNESS,
                constants.RagasMetricsConstants.CONTEXT_RECALL,
            ]
            metrics_to_run = [
                m for m in self.metrics if m.name not in metrics_that_need_ground_truth
            ]

        if not metrics_to_run:
            return {}

        data = Dataset.from_dict(dataset_dict)

        result = evaluate(data, metrics=metrics_to_run, llm=self.llm, raise_exceptions=False)

        metrics_dict = {}
        if result:
            for metric_name, score in result.items():
                if isinstance(score, list) and score:
                    metrics_dict[metric_name] = round(score[0], 2)

        end_time = time.time()
        self.time_taken = end_time - start_time
        return metrics_dict

    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken