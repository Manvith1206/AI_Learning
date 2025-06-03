from typing import Dict, List, Any, Optional
from app.domain.models import EvaluationResult
from app.domain.services import RAGService


class EvaluationUseCase:
    """Application use case for RAG evaluation"""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.last_evaluation: Optional[EvaluationResult] = None
    
    def evaluate_last_query(self, ground_truth: str) -> EvaluationResult:
        """Evaluate the last query against a ground truth"""
        # Evaluate using RAG service
        evaluation_result = self.rag_service.evaluate(ground_truth)
        
        # Store for later reference
        self.last_evaluation = evaluation_result
        
        return evaluation_result
    
    def get_last_evaluation(self) -> Optional[EvaluationResult]:
        """Get the last evaluation result"""
        return self.last_evaluation
    
    def calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate an overall score from individual metrics"""
        if not metrics:
            return 0.0
        
        return sum(metrics.values()) / len(metrics)
