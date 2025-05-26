from abc import ABC, abstractmethod
from typing import List, Dict, Any, Protocol
import re

from .base_evaluator import BaseEvaluator
import streamlit as st
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import rag_modular.Common.RAG_Constants as constants

# --- LLM Service Interface ---
class LLMService(Protocol):
    """Protocol for a generic LLM service."""

    def evaluate_statement(self, statement: str, context: str, prompt_template: str):
        """Evaluates if a statement is supported by the given context using a specific prompt."""
        ...

    def generate_text(self, prompt: str):
        """Generates text based on a given prompt."""
        ...

    def generate_questions(self, answer: str, prompt_template: str, num_questions: int = 3):
        """Generates questions for which the given answer would be appropriate."""
        ...

    def calculate_similarity(self, text1: str, text2: str):
        """Calculates semantic similarity between two texts (e.g., using embeddings and cosine similarity)."""
        # This might involve calling an embedding model and then a similarity function.
        # For simplicity, this might be a direct call if the LLM service supports it,
        # or it might need a separate embedding component.
        ...

# --- Concrete LLM Services ---
class GeminiLLMService(LLMService):
    """LLM Service implementation using Google Gemini."""
    def __init__(self, api_key: str, generative_model_name: str = "gemini-1.5-flash-latest", embedding_model_name: str = constants.GeminiEmbedModels.GEMINI_TEXT_EMBED_MODEL.value):
        genai.configure(api_key=api_key)
        self.generative_model = genai.GenerativeModel(generative_model_name)
        self.embedding_model_name = embedding_model_name # Store for embed_content
        print(f"GeminiLLMService initialized with generative model: {generative_model_name} and embedding model: {embedding_model_name}")

    def evaluate_statement(self, statement: str, context: str, prompt_template: str):
        prompt = prompt_template.format(statement=statement, context=context)
        try:
            response = self.generative_model.generate_content(prompt)
            decision = response.text.strip().lower()
            return "yes" in decision
        except Exception as e:
            print(f"Error during Gemini statement evaluation: {e}")
            return False # Default to false on error

    def generate_text(self, prompt: str):
        try:
            response = self.generative_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error during Gemini text generation: {e}")
            return "" # Default to empty string on error

    def generate_questions(self, answer: str, prompt_template: str, num_questions: int = 3):
        prompt = prompt_template.format(answer=answer, num_questions=num_questions)
        try:
            response_text = self.generate_text(prompt)
            # Assuming the LLM returns questions separated by newlines
            questions = [q.strip() for q in response_text.split('\n') if q.strip()]
            return questions[:num_questions]
        except Exception as e:
            print(f"Error during Gemini question generation: {e}")
            return []

    def calculate_similarity(self, text1: str, text2: str):
        try:
            # Gemini's embed_content can take a list of texts
            result = genai.embed_content(model=self.embedding_model_name, content=[text1, text2])
            embedding1 = np.array(result['embedding'][0]).reshape(1, -1)
            embedding2 = np.array(result['embedding'][1]).reshape(1, -1)
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error during Gemini similarity calculation: {e}")
            return 0.0 # Default to 0.0 on error

# --- Concrete LLM Service (Example: OpenAI) ---
# You would implement concrete classes like OpenAIService, GeminiService, etc.
# For now, we'll assume a mock or a conceptual implementation.

class OpenAILLMService(LLMService):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        # In a real scenario, initialize the OpenAI client here
        # from openai import OpenAI
        # self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        print(f"OpenAILLMService initialized with model: {model_name}")

    def evaluate_statement(self, statement: str, context: str, prompt_template: str):
        # prompt = prompt_template.format(statement=statement, context=context)
        # response = self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=[{"role": "system", "content": "You are an expert verifier."},
        #               {"role": "user", "content": prompt}],
        #     max_tokens=10,
        #     temperature=0
        # )
        # decision = response.choices[0].message.content.strip().lower()
        # return "yes" in decision
        print(f"Mock OpenAI: Evaluating statement '{statement}' against context. Assuming 'yes'.")
        return True # Mock implementation

    def generate_text(self, prompt: str):
        # response = self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=150,
        #     temperature=0.7
        # )
        # return response.choices[0].message.content.strip()
        print(f"Mock OpenAI: Generating text for prompt. Returning placeholder.")
        return "Mock generated text." # Mock implementation

    def generate_questions(self, answer: str, prompt_template: str, num_questions: int = 3):
        # prompt = prompt_template.format(answer=answer, num_questions=num_questions)
        # response = self.generate_text(prompt)
        # # Assuming the LLM returns questions separated by newlines or a specific format
        # questions = [q.strip() for q in response.split('\n') if q.strip()]
        # return questions[:num_questions]
        print(f"Mock OpenAI: Generating {num_questions} questions for answer. Returning placeholders.")
        return [f"Mock question {i+1} for the answer." for i in range(num_questions)] # Mock

    def calculate_similarity(self, text1: str, text2: str):
        # This would typically involve getting embeddings for text1 and text2
        # and then computing cosine similarity. For simplicity, returning a mock value.
        # from sklearn.metrics.pairwise import cosine_similarity
        # from some_embedding_service import get_embedding
        # emb1 = get_embedding(text1)
        # emb2 = get_embedding(text2)
        # return cosine_similarity(emb1, emb2)[0][0]
        print(f"Mock OpenAI: Calculating similarity between texts. Returning placeholder value.")
        return 0.85 # Mock implementation

# --- Evaluation Metric Base Class ---
class EvaluationMetric(ABC):
    """Abstract base class for an evaluation metric."""
    metric_name: str = "base_metric"

    def __init__(self, llm_service: LLMService):
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

    def __init__(self, llm_service: LLMService, prompt_template: str = None):
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

    def __init__(self, llm_service: LLMService, prompt_template: str = None):
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

    def __init__(self, llm_service: LLMService, prompt_template: str = None):
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
        "Generate {num_questions} diverse questions for which the following answer would be a suitable and relevant response. "
        "Return each question on a new line.\n\n"
        "Answer:\n{answer}"
    )

    def __init__(self, llm_service: LLMService, question_gen_prompt: str = None, num_generated_questions: int = 3):
        super().__init__(llm_service)
        self.question_gen_prompt_template = question_gen_prompt or self.default_question_generation_prompt_template
        self.num_generated_questions = num_generated_questions

    def calculate(self, question: str, answer: str, contexts: List[str], ground_truth: str = None):
        if not question or not answer:
            return 0.0

        generated_questions = self.llm_service.generate_questions(
            answer, 
            self.question_gen_prompt_template,
            self.num_generated_questions
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

    def __init__(self, metrics: List[EvaluationMetric], llm_service: LLMService = None):
        """
        Initialize with a list of evaluation metrics and an LLM service.
        If llm_service is None, it implies metrics are pre-configured with their own LLM services.
        If llm_service is provided, it can be used as a default for metrics if they don't have one.
        However, current metric design requires LLMService at metric initialization.
        """
        self.metrics = metrics
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
        
        print("LLMService: ", config)
        print("Custom Evalautor")
        print("Question: ", question)
        print("Answer: ", answer)
        print("Contexts: ", contexts)
        print("GroundTruths: ", ground_truths)
        for metric in self.metrics:
            try:
                score = metric.calculate(question, answer, contexts, ground_truths)
                results[metric.metric_name] = score
            except Exception as e:
                print(f"Error calculating metric {metric.metric_name}: {e}")
                results[metric.metric_name] = None # Or 0.0, or handle as per requirement
        return results


# # --- Example Usage (for testing purposes) ---
# if __name__ == '__main__':
#     # Configure API keys (ensure these are set in Streamlit secrets or environment variables)
#     # For Gemini (ensure st.secrets has constants.GEMINI_API_KEY)
#     try:
#         gemini_api_key = st.secrets[constants.GEMINI_API_KEY]
#     except (AttributeError, KeyError):
#         gemini_api_key = "YOUR_GEMINI_API_KEY_FALLBACK" # Fallback if secrets not found, for local testing
#         print("Warning: Gemini API key not found in st.secrets. Using fallback.")

#     # This is a mock/conceptual LLM service for demonstration.
#     # We will now use the GeminiLLMService if the API key is available.
#     # If you want to test OpenAI, ensure 'openai_api_key' is configured similarly.

#     # In a real application, you'd use a concrete implementation like OpenAILLMService,
#     # GeminiLLMService, AnthropicLLMService, etc., with actual API keys and client setup.
#     # Initialize the LLM service (e.g., Gemini)
#     if gemini_api_key and gemini_api_key != "YOUR_GEMINI_API_KEY_FALLBACK":
#         llm_service = GeminiLLMService(api_key=gemini_api_key)
#         print("Using GeminiLLMService for evaluation example.")
#     else:
#         print("Gemini API key not configured or fallback is active. Using Mock OpenAILLMService for example.")
#         # Fallback to mock OpenAI if Gemini key is not set up for the example
#         llm_service = OpenAILLMService(api_key="DUMMY_KEY_FOR_MOCK_OPENAI")

#     # Initialize metrics with the chosen LLM service
#     faithfulness_metric = FaithfulnessMetric(llm_service=llm_service)
#     context_precision_metric = ContextPrecisionMetric(llm_service=llm_service)
#     context_recall_metric = ContextRecallMetric(llm_service=llm_service)
#     answer_relevancy_metric = AnswerRelevancyMetric(llm_service=llm_service)

#     # Initialize CustomEvaluator with the list of metrics
#     custom_eval = CustomEvaluator(
#         metrics=[
#             faithfulness_metric,
#             context_precision_metric,
#             context_recall_metric,
#             answer_relevancy_metric
#         ]
#     )

#     # Sample data for evaluation
#     sample_question = "What is the capital of France?"
#     sample_answer = "The capital of France is Paris."
#     sample_contexts = [
#         "France is a country in Western Europe. Paris is its capital and largest city.",
#         "Berlin is the capital of Germany.",
#         "The Eiffel Tower is a famous landmark in Paris."
#     ]
#     sample_ground_truth = "Paris is the capital city of France."

#     print("\nRunning custom evaluation...")
#     evaluation_results = custom_eval.evaluate(
#         question=sample_question,
#         answer=sample_answer,
#         contexts=sample_contexts,
#         ground_truths=sample_ground_truth
#     )

#     print("\nEvaluation Results:")
#     if evaluation_results:
#         for metric_name, score in evaluation_results.items():
#             print(f"  {metric_name}: {score:.2f}" if score is not None else f"  {metric_name}: Error/Unavailable")
#     else:
#         print("  No evaluation results produced.")
        
#     # Example with a custom prompt for Faithfulness
#     custom_faith_prompt = (
#         "Based *only* on the provided text: '{context}', is the claim '{statement}' verifiably true? Answer 'yes' or 'no'."
#     )
#     faithfulness_metric_custom_prompt = FaithfulnessMetric(
#         llm_service=llm_service, 
#         prompt_template=custom_faith_prompt
#     )
#     custom_eval_single_metric = CustomEvaluator(metrics=[faithfulness_metric_custom_prompt])
    
#     print("\nRunning custom evaluation for single metric (Faithfulness with custom prompt)...")
#     single_metric_results = custom_eval_single_metric.evaluate(
#         question=sample_question,
#         answer=sample_answer,
#         contexts=sample_contexts,
#         ground_truths=sample_ground_truth
#     )
    
#     if single_metric_results and FaithfulnessMetric.metric_name in single_metric_results:
#         score = single_metric_results[FaithfulnessMetric.metric_name]
#         print(f"  Faithfulness with custom prompt: {score:.2f}" if score is not None else "  Faithfulness with custom prompt: Error/Unavailable")
#     else:
#         print("  Could not retrieve Faithfulness score with custom prompt.")
