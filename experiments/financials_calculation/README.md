# Financials Calculation Experiment

## Overview

This experiment evaluates different LLM models on their ability to extract or calculate **Cost of Revenue** from structured XBRL financial facts. Models are tested on their capability to either directly extract values from specific XBRL concepts, apply financial formulas (e.g., Revenue - Gross Profit), or impute values using industry-specific tags when direct methods are unavailable.

## Evaluation Metrics

The experiment uses comprehensive regression metrics to evaluate model performance:

- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and ground truth
- **RMSE (Root Mean Squared Error)**: Square root of average squared errors, penalizes larger errors more heavily
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error relative to ground truth values
- **R² (R-squared)**: Coefficient of determination indicating how well predictions explain variance in ground truth
- **Accuracy within thresholds**: Percentage of predictions within 5%, 10%, and 20% of ground truth values
- **Cost & Duration**: Average API cost and response time per prediction

## How to Run

Execute the experiment using the main script:

```bash
python experiments/financials_calculation/main.py
```

This will:
1. Load the financials calculation dataset
2. Run predictions across enabled LLM models (OpenAI, Anthropic, Gemini)
3. Evaluate results using regression metrics
4. Save both evaluation metrics and raw predictions to timestamped JSON files
5. Display results summary in the console

## Output Files

- `financials_calculation_results_[timestamp].json`: Aggregated evaluation metrics
- `financials_calculation_predictions_[timestamp].json`: Raw model predictions for manual validation 