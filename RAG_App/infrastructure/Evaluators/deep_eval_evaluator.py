import time
from infrastructure.evaluators.base_evaluator import BaseEvaluator
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
import infrastructure.common.rag_constants as constants
from deepeval.models.llms import gemini_model

class DeepEval(BaseEvaluator):
    def __init__(self, api_key, metrics=None):
        model = gemini_model.GeminiModel(model_name=constants.GeminiLLMModel.GEMINI_FLASH.value, api_key=api_key)
        self.metrics = metrics or [
            FaithfulnessMetric(threshold=0.7, model=model, include_reason=True),
            AnswerRelevancyMetric(threshold=0.7, model=model, include_reason=True),
            ContextualPrecisionMetric(threshold=0.7, model=model, include_reason=True),
            ContextualRecallMetric(threshold=0.7, model=model, include_reason=True)
        ]
        self.time_taken = 0
        self.cost = 0
    def evaluate(self, question, answer, contexts, ground_truths=None):
        # Replace this with the actual output from your LLM application
        start_time = time.time()  # Start timing
        actual_output = answer

        # Replace this with the actual retrieved context from your RAG pipeline
        retrieval_context = contexts
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
            expected_output=ground_truths
        )

        result = evaluate(test_cases=[test_case], metrics=self.metrics)

        metrics_dict = {}

        curr_cost_value = 0
        for test_result in result.test_results:
            for metric_result in test_result.metrics_data:
                
                if metric_result.name == constants.DEEP_EVAL_FAITHFULNESS:
                    metrics_dict[constants.FAITHFULNESS] = round(metric_result.score, 2)
                elif metric_result.name == constants.DEEP_EVAL_CONTEXT_PRECISION:
                    metrics_dict[constants.CONTEXT_PRECISION] = round(metric_result.score, 2)
                elif metric_result.name == constants.DEEP_EVAL_CONTEXT_RECALL:
                    metrics_dict[constants.CONTEXT_RECALL] = round(metric_result.score, 2)
                elif metric_result.name == constants.DEEP_EVAL_ANSWER_RELEVANCY:
                    metrics_dict[constants.ANSWER_RELEVANCY] = round(metric_result.score, 2)
                curr_cost_value += metric_result.evaluation_cost
        end_time = time.time()  # End timing
        self.time_taken = end_time - start_time
        self.cost = curr_cost_value
        return metrics_dict
    
    def get_cost_and_time_taken(self):
        return self.cost, self.time_taken
    