import math
from experiments.common.models import RegressionEvaluationMetrics, RegressionComparisonResults
from experiments.financials_calculation.experiment import ExperimentResults, ModelResults


class FinancialsCalculationJudge:

    def evaluate(self, results: ExperimentResults) -> RegressionComparisonResults:
        """Evaluate all model results and return comprehensive regression metrics."""
        evaluation_results = RegressionComparisonResults()
        
        # Evaluate each model if results exist
        if results.openai:
            evaluation_results.openai = self._evaluate_model(results.openai)
        
        if results.anthropic:
            evaluation_results.anthropic = self._evaluate_model(results.anthropic)
            
        if results.gemini:
            evaluation_results.gemini = self._evaluate_model(results.gemini)
        
        if results.kimi:
            evaluation_results.kimi = self._evaluate_model(results.kimi)
        
        if results.deepseek:
            evaluation_results.deepseek = self._evaluate_model(results.deepseek)
        
        # Find best performing models by different metrics
        evaluation_results.best_mae_model = self._find_best_model_by_metric(
            evaluation_results, "mean_absolute_error", lower_is_better=True
        )
        evaluation_results.best_rmse_model = self._find_best_model_by_metric(
            evaluation_results, "root_mean_squared_error", lower_is_better=True
        )
        evaluation_results.best_r2_model = self._find_best_model_by_metric(
            evaluation_results, "r_squared", lower_is_better=False
        )
        evaluation_results.best_accuracy_5pct_model = self._find_best_model_by_metric(
            evaluation_results, "accuracy_within_5_percent", lower_is_better=False
        )
        
        return evaluation_results

    def _evaluate_model(self, model_results: ModelResults) -> RegressionEvaluationMetrics:
        """Evaluate a single model's regression performance."""
        predictions = model_results.predictions
        
        if not predictions:
            return self._empty_metrics(model_results)
        
        # Extract predictions and ground truths
        y_pred = [p.prediction for p in predictions]
        y_true = [p.ground_truth for p in predictions]
        
        # Filter out any None values
        valid_pairs = [(pred, true) for pred, true in zip(y_pred, y_true) 
                      if pred is not None and true is not None]
        
        if not valid_pairs:
            return self._empty_metrics(model_results)
        
        y_pred_valid = [pair[0] for pair in valid_pairs]
        y_true_valid = [pair[1] for pair in valid_pairs]
        n = len(valid_pairs)
        
        # Calculate regression metrics
        mae = sum(abs(pred - true) for pred, true in valid_pairs) / n
        mse = sum((pred - true) ** 2 for pred, true in valid_pairs) / n
        rmse = math.sqrt(mse)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Handle division by zero by excluding zero ground truth values
        mape_values = []
        for pred, true in valid_pairs:
            if true != 0:
                mape_values.append(abs((pred - true) / true))
        mape = (sum(mape_values) / len(mape_values) * 100) if mape_values else float('inf')
        
        # Calculate R² score
        y_mean = sum(y_true_valid) / n
        ss_tot = sum((true - y_mean) ** 2 for true in y_true_valid)
        ss_res = sum((true - pred) ** 2 for pred, true in valid_pairs)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        # Calculate accuracy within percentage thresholds
        accuracy_5pct = self._calculate_percentage_accuracy(valid_pairs, 0.05)
        accuracy_10pct = self._calculate_percentage_accuracy(valid_pairs, 0.10)
        accuracy_20pct = self._calculate_percentage_accuracy(valid_pairs, 0.20)
        
        return RegressionEvaluationMetrics(
            model_provider=model_results.model_provider,
            model_name=model_results.model_name,
            total_predictions=n,
            mean_absolute_error=mae,
            mean_squared_error=mse,
            root_mean_squared_error=rmse,
            mean_absolute_percentage_error=mape,
            r_squared=r_squared,
            accuracy_within_5_percent=accuracy_5pct,
            accuracy_within_10_percent=accuracy_10pct,
            accuracy_within_20_percent=accuracy_20pct,
            average_cost=model_results.average_cost,
            average_duration=model_results.average_duration
        )

    def _calculate_percentage_accuracy(self, valid_pairs: list[tuple[float, float]], threshold: float) -> float:
        """Calculate the percentage of predictions within a given percentage threshold of ground truth."""
        if not valid_pairs:
            return 0.0
        
        within_threshold = 0
        for pred, true in valid_pairs:
            if true == 0:
                # Handle zero ground truth case
                if pred == 0:
                    within_threshold += 1
            else:
                percentage_error = abs((pred - true) / true)
                if percentage_error <= threshold:
                    within_threshold += 1
        
        return (within_threshold / len(valid_pairs)) * 100

    def _empty_metrics(self, model_results: ModelResults) -> RegressionEvaluationMetrics:
        """Return empty metrics for models with no valid predictions."""
        return RegressionEvaluationMetrics(
            model_provider=model_results.model_provider,
            model_name=model_results.model_name,
            total_predictions=0,
            mean_absolute_error=float('inf'),
            mean_squared_error=float('inf'),
            root_mean_squared_error=float('inf'),
            mean_absolute_percentage_error=float('inf'),
            r_squared=0.0,
            accuracy_within_5_percent=0.0,
            accuracy_within_10_percent=0.0,
            accuracy_within_20_percent=0.0,
            average_cost=model_results.average_cost,
            average_duration=model_results.average_duration
        )

    def _find_best_model_by_metric(self, results: RegressionComparisonResults, metric: str, lower_is_better: bool = True) -> str | None:
        """Find the model with the best performance for a given metric."""
        models = []
        
        if results.openai:
            models.append(("openai", getattr(results.openai, metric)))
        if results.anthropic:
            models.append(("anthropic", getattr(results.anthropic, metric)))
        if results.gemini:
            models.append(("gemini", getattr(results.gemini, metric)))
        if results.kimi:
            models.append(("kimi", getattr(results.kimi, metric)))
        if results.deepseek:
            models.append(("deepseek", getattr(results.deepseek, metric)))
        
        if not models:
            return None
        
        # Filter out infinite values
        valid_models = [(name, value) for name, value in models if not math.isinf(value)]
        if not valid_models:
            return None
            
        # Find model with best metric value
        if lower_is_better:
            best_model = min(valid_models, key=lambda x: x[1])
        else:
            best_model = max(valid_models, key=lambda x: x[1])
        
        return best_model[0]

    def print_evaluation_summary(self, results: RegressionComparisonResults) -> None:
        """Print a formatted summary of regression evaluation results."""
        print("\n" + "="*70)
        print("FINANCIALS CALCULATION EVALUATION RESULTS")
        print("="*70)
        
        models = [
            ("OpenAI", results.openai),
            ("Anthropic", results.anthropic), 
            ("Gemini", results.gemini),
            ("Kimi", results.kimi),
            ("DeepSeek", results.deepseek)
        ]
        
        for provider_name, metrics in models:
            if metrics:
                print(f"\n{provider_name} ({metrics.model_name}):")
                print(f"  Total Predictions: {metrics.total_predictions}")
                print(f"  MAE:              ${metrics.mean_absolute_error:,.0f}")
                print(f"  RMSE:             ${metrics.root_mean_squared_error:,.0f}")
                
                if not math.isinf(metrics.mean_absolute_percentage_error):
                    print(f"  MAPE:             {metrics.mean_absolute_percentage_error:.2f}%")
                else:
                    print(f"  MAPE:             N/A (division by zero)")
                
                print(f"  R²:               {metrics.r_squared:.3f}")
                print(f"  Accuracy (±5%):   {metrics.accuracy_within_5_percent:.1f}%")
                print(f"  Accuracy (±10%):  {metrics.accuracy_within_10_percent:.1f}%")
                print(f"  Accuracy (±20%):  {metrics.accuracy_within_20_percent:.1f}%")
                print(f"  Avg Cost:         ${metrics.average_cost:.4f}")
                print(f"  Avg Duration:     {metrics.average_duration:.2f}s")
        
        print(f"\nBest Models:")
        print(f"  Lowest MAE:       {results.best_mae_model}")
        print(f"  Lowest RMSE:      {results.best_rmse_model}")
        print(f"  Highest R²:       {results.best_r2_model}")
        print(f"  Best ±5% Accuracy: {results.best_accuracy_5pct_model}")
        print("="*70)
