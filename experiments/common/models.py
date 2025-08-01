from pydantic import BaseModel

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


class RegressionEvaluationMetrics(BaseModel):
    """Evaluation metrics for regression models."""
    model_provider: str
    model_name: str
    total_predictions: int
    mean_absolute_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    mean_absolute_percentage_error: float
    r_squared: float
    accuracy_within_5_percent: float  # Percentage of predictions within 5% of ground truth
    accuracy_within_10_percent: float  # Percentage of predictions within 10% of ground truth
    accuracy_within_20_percent: float  # Percentage of predictions within 20% of ground truth
    average_cost: float
    average_duration: float


class ComparisonResults(BaseModel):
    """Complete evaluation results comparing all models."""
    openai: ModelEvaluationMetrics | None = None
    anthropic: ModelEvaluationMetrics | None = None
    gemini: ModelEvaluationMetrics | None = None
    kimi: ModelEvaluationMetrics | None = None
    deepseek: ModelEvaluationMetrics | None = None
    best_accuracy_model: str | None = None
    best_f1_model: str | None = None


class RegressionComparisonResults(BaseModel):
    """Complete regression evaluation results comparing all models."""
    openai: RegressionEvaluationMetrics | None = None
    anthropic: RegressionEvaluationMetrics | None = None
    gemini: RegressionEvaluationMetrics | None = None
    kimi: RegressionEvaluationMetrics | None = None
    deepseek: RegressionEvaluationMetrics | None = None
    best_mae_model: str | None = None  # Best Mean Absolute Error
    best_rmse_model: str | None = None  # Best Root Mean Squared Error
    best_r2_model: str | None = None  # Best R² score
    best_accuracy_5pct_model: str | None = None  # Best accuracy within 5%
