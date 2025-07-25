from pydantic import BaseModel
from typing import Optional
from experiments.red_flag_detection.experiment import ExperimentResults, ModelResults, LLMPredictionResult


class ModelEvaluationMetrics(BaseModel):
    """Evaluation metrics for a single model."""
    model_provider: str
    model_name: str
    total_predictions: int
    correct_predictions: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    average_cost: float
    average_duration: float


class ComparisonResults(BaseModel):
    """Complete evaluation results comparing all models."""
    openai: Optional[ModelEvaluationMetrics] = None
    anthropic: Optional[ModelEvaluationMetrics] = None
    gemini: Optional[ModelEvaluationMetrics] = None
    best_accuracy_model: Optional[str] = None
    best_f1_model: Optional[str] = None


class RedFlagDetectionJudge:

    def evaluate(self, results: ExperimentResults) -> ComparisonResults:
        """Evaluate all model results and return comprehensive metrics."""
        evaluation_results = ComparisonResults()
        
        # Evaluate each model if results exist
        if results.openai:
            evaluation_results.openai = self._evaluate_model(results.openai)
        
        if results.anthropic:
            evaluation_results.anthropic = self._evaluate_model(results.anthropic)
            
        if results.gemini:
            evaluation_results.gemini = self._evaluate_model(results.gemini)
        
        # Find best performing models
        evaluation_results.best_accuracy_model = self._find_best_model_by_metric(
            evaluation_results, "accuracy"
        )
        evaluation_results.best_f1_model = self._find_best_model_by_metric(
            evaluation_results, "f1_score"
        )
        
        return evaluation_results

    def _evaluate_model(self, model_results: ModelResults) -> ModelEvaluationMetrics:
        """Evaluate a single model's performance."""
        predictions = model_results.predictions
        
        # Calculate confusion matrix components
        tp = sum(1 for p in predictions if p.prediction == True and p.ground_truth == True)
        fp = sum(1 for p in predictions if p.prediction == True and p.ground_truth == False)
        tn = sum(1 for p in predictions if p.prediction == False and p.ground_truth == False)
        fn = sum(1 for p in predictions if p.prediction == False and p.ground_truth == True)
        
        # Calculate metrics
        total = len(predictions)
        correct = tp + tn
        accuracy = correct / total if total > 0 else 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return ModelEvaluationMetrics(
            model_provider=model_results.model_provider,
            model_name=model_results.model_name,
            total_predictions=total,
            correct_predictions=correct,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            average_cost=model_results.average_cost,
            average_duration=model_results.average_duration
        )

    def _find_best_model_by_metric(self, results: ComparisonResults, metric: str) -> Optional[str]:
        """Find the model with the best performance for a given metric."""
        models = []
        
        if results.openai:
            models.append(("openai", getattr(results.openai, metric)))
        if results.anthropic:
            models.append(("anthropic", getattr(results.anthropic, metric)))
        if results.gemini:
            models.append(("gemini", getattr(results.gemini, metric)))
        
        if not models:
            return None
            
        # Find model with highest metric value
        best_model = max(models, key=lambda x: x[1])
        return best_model[0]

    def print_evaluation_summary(self, results: ComparisonResults) -> None:
        """Print a formatted summary of evaluation results."""
        print("\n" + "="*60)
        print("RED FLAG DETECTION EVALUATION RESULTS")
        print("="*60)
        
        models = [
            ("OpenAI", results.openai),
            ("Anthropic", results.anthropic), 
            ("Gemini", results.gemini)
        ]
        
        for provider_name, metrics in models:
            if metrics:
                print(f"\n{provider_name} ({metrics.model_name}):")
                print(f"  Accuracy:   {metrics.accuracy:.3f} ({metrics.correct_predictions}/{metrics.total_predictions})")
                print(f"  Precision:  {metrics.precision:.3f}")
                print(f"  Recall:     {metrics.recall:.3f}")  
                print(f"  F1 Score:   {metrics.f1_score:.3f}")
                print(f"  TP: {metrics.true_positives}, FP: {metrics.false_positives}, TN: {metrics.true_negatives}, FN: {metrics.false_negatives}")
        
        print(f"\nBest Accuracy: {results.best_accuracy_model}")
        print(f"Best F1 Score: {results.best_f1_model}")
        print("="*60)