from abc import ABC, abstractmethod
import time
from typing import List, Dict, Any, Protocol
import re

from .base_evaluator import BaseEvaluator
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import infrastructure.Common.RAG_Constants as constants
from infrastructure.Evaluators.LLM_Evaluation_Service import LLM_Evaluation_Service

# --- Evaluation Metric Base Class ---
class EvaluationMetric(ABC):
    """Abstract base class for an evaluation metric."""
    metric_name: str = "base_metric"

    def __init__(self, llm_service: LLM_Evaluation_Service):
        self.llm_service = llm_service

    @abstractmethod
    def calculate(self, question: str, answer: str, contexts: List[str], ground_truth: str = None):
        """Calculates the metric score."""
        pass

# --- Concrete Metric Implementations ---

class FaithfulnessMetric(EvaluationMetric):
    metric_name: str = "custom_faithfulness"
    default_prompt_template: str = (
        "Given the following context, please determine if the statement below is directly supported by the information "
        "in the context. Respond with only 'yes' or 'no'.\n\n"
        "Context:\n{context}\n\n"
        "Statement:\n{statement}"
    )

    def __init__(self, llm_service: LLM_Evaluation_Service, prompt_template: str = None):
        super().__init__(llm_service)
        self.prompt_template = prompt_template or self.default_prompt_template

    def _extract_statements(self, text: str) -> List[str]:
        # Simple statement extraction: split by sentences. More sophisticated methods can be used.
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
        return [s.strip() for s in sentences if s.strip()]

    def calculate(self, question: str, answer: str, contexts: List[str], ground_truth: str = None):
        if not answer or not contexts:
            return 0.0
        
        statements = self._extract_statements(answer)
        if not statements:
            return 0.0

        supported_statements = 0
        # Concatenate all contexts into a single string for verification
        full_context = "\n".join(contexts)

        for stmt in statements:
            if self.llm_service.evaluate_statement(stmt, full_context, self.prompt_template):
                supported_statements += 1
        
        return supported_statements / len(statements)

class ContextPrecisionMetric(EvaluationMetric):
    metric_name: str = "custom_context_precision"
    default_prompt_template: str = (
        "Given the question and the following context chunk, is this context chunk relevant to answering the question? "
        "Respond with only 'yes' or 'no'.\n\n"
        "Question:\n{question}\n\n"
        "Context Chunk:\n{context_chunk}"
    )

    def __init__(self, llm_service: LLM_Evaluation_Service, prompt_template: str = None):
        super().__init__(llm_service)
        self.prompt_template = prompt_template or self.default_prompt_template

    def calculate(self, question: str, answer: str, contexts: List[str], ground_truth: str = None):
        if not question or not contexts:
            return 0.0

        relevant_chunks_at_k = 0
        precision_sum = 0.0
        relevant_chunks_count_for_weight = 0

        for k, chunk in enumerate(contexts, 1):
            is_relevant = self.llm_service.evaluate_statement(
                statement=question,  # Using question as the 'statement' to check against chunk
                context=chunk, 
                prompt_template=self.prompt_template.format(question=question, context_chunk=chunk)
            )
            if is_relevant:
                relevant_chunks_at_k += 1
                relevant_chunks_count_for_weight +=1
                precision_sum += (relevant_chunks_at_k / k) # Precision@k * 1 (weight for relevant)
            # else: Precision@k * 0 (weight for irrelevant), so no need to add to precision_sum
            
        if relevant_chunks_count_for_weight == 0:
             return 0.0 # Avoid division by zero if no chunks are deemed relevant

        # Weighted average: Sum(Precision@k * rel_k) / Sum(rel_k)
        # rel_k is 1 if chunk at k is relevant, 0 otherwise.
        # Sum(rel_k) is effectively relevant_chunks_count_for_weight
        return precision_sum / relevant_chunks_count_for_weight

class ContextRecallMetric(EvaluationMetric):
    metric_name: str = "custom_context_recall"
    default_prompt_template: str = (
        "Given the following context, can the statement from the ground truth answer be inferred from this context? "
        "Respond with only 'yes' or 'no'.\n\n"
        "Context:\n{context}\n\n"
        "Ground Truth Statement:\n{statement}"
    )

    def __init__(self, llm_service: LLM_Evaluation_Service, prompt_template: str = None):
        super().__init__(llm_service)
        self.prompt_template = prompt_template or self.default_prompt_template

    def _extract_statements(self, text: str):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
        return [s.strip() for s in sentences if s.strip()]

    def calculate(self, question: str, answer: str, contexts: List[str], ground_truth: str = None):
        if not ground_truth or not contexts:
            # This metric requires ground truth
            return 0.0 # Or raise an error, or return None/NaN
        
        gt_statements = self._extract_statements(ground_truth)
        if not gt_statements:
            return 0.0

        attributable_statements = 0
        full_context = "\n".join(contexts)

        for stmt in gt_statements:
            if self.llm_service.evaluate_statement(stmt, full_context, self.prompt_template):
                attributable_statements += 1
        
        return attributable_statements / len(gt_statements)

class AnswerRelevancyMetric(EvaluationMetric):
    metric_name: str = "custom_answer_relevancy"
    default_question_generation_prompt_template: str = (
        """
        Original Question: {original_question}
        Answer: {answer}

        Generate {num_questions} questions that are similar to the original question and would have the same answer.
        The generated questions should:
        1. Preserve the main intent and meaning of the original question
        2. Use similar key terms and entities  
        3. Have the same question type (what, how, why, etc.)
        4. Be answerable by the same response

        Output Format:
        Generated Questions:
        1. Question 1
        """
    )

    def __init__(self, llm_service: LLM_Evaluation_Service, question_gen_prompt: str = None, num_generated_questions: int = 3):
        super().__init__(llm_service)
        self.question_gen_prompt_template = question_gen_prompt or self.default_question_generation_prompt_template
        self.num_generated_questions = num_generated_questions

    def calculate(self, question: str, answer: str, contexts: List[str], ground_truth: str = None):
        if not question or not answer:
            return 0.0

        generated_questions = self.llm_service.generate_questions(
            answer=answer, 
            original_question=question,
            prompt_template=self.question_gen_prompt_template,
            num_questions=self.num_generated_questions
        )
        
        if not generated_questions:
            return 0.0

        total_similarity = 0.0
        for gen_q in generated_questions:
            total_similarity += self.llm_service.calculate_similarity(question, gen_q)
        
        return total_similarity / len(generated_questions)

# --- Custom Evaluator ---
class CustomEvaluator(BaseEvaluator):
    """Evaluator that uses custom metrics and LLM services."""

    def __init__(self, metrics: List[EvaluationMetric], llm_service: LLM_Evaluation_Service = None):
        """
        Initialize with a list of evaluation metrics and an LLM service.
        If llm_service is None, it implies metrics are pre-configured with their own LLM services.
        If llm_service is provided, it can be used as a default for metrics if they don't have one.
        However, current metric design requires LLMService at metric initialization.
        """
        self.metrics = metrics
        self.cost = 0
        self.time_taken = 0
        # Each metric should be initialized with an LLM service already.
        # self.llm_service = llm_service 

    def evaluate(self, question: str, answer: str, contexts: List[str], ground_truths: str = None):
        """
        Evaluate using the configured custom metrics.
        
        Args:
            question: The query/question asked
            answer: The generated answer
            contexts: The contexts used to generate the answer (list of strings)
            ground_truths: Optional ground truth answer (single string)
            
        Returns:
            Dictionary of evaluation scores, with metric names as keys.
        """
        
        results = {}
        start_time = time.time()
        
        for metric in self.metrics:
            try:
                score = metric.calculate(question, answer, contexts, ground_truths)
                results[metric.metric_name] = score
            except Exception as e:
                print(f"Error calculating metric {metric.metric_name}: {e}")
                results[metric.metric_name] = None # Or 0.0, or handle as per requirement

        end_time = time.time()
        self.time_taken = end_time - start_time
        return results
    
    def get_cost_and_time_taken(self):
        return 0,self.time_taken