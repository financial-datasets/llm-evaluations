# Red Flag Detection Experiment

## Overview

This experiment evaluates different LLM models on their ability to detect financial red flags in companies based on financial metrics. Models analyze structured financial data to determine whether a company shows signs of financial risk (declining earnings, high debt, poor liquidity) or appears financially healthy overall. This is a binary classification task where models must make a true/false decision about whether red flags are present.

## Evaluation Metrics

The experiment uses standard binary classification metrics to evaluate model performance:

- **Accuracy**: Overall percentage of correct predictions (both red and green flag companies)
- **Precision**: Of companies flagged as risky, what percentage actually have red flags
- **Recall**: Of companies that actually have red flags, what percentage were correctly identified
- **F1 Score**: Harmonic mean of precision and recall, balancing both metrics
- **Confusion Matrix**: Complete breakdown of True/False Positives and True/False Negatives
- **Cost & Duration**: Average API cost and response time per prediction

## How to Run

Execute the experiment using the main script:

```bash
python experiments/red_flag_detection/main.py
```

This will:
1. Load the red flag detection dataset (containing both risky and healthy companies)
2. Run binary classification predictions across all enabled LLM models
3. Evaluate results using classification metrics
4. Save evaluation results to a timestamped JSON file
5. Display results summary in the console

## Output Files

- `red_flag_detection_results_[timestamp].json`: Complete evaluation metrics and model comparison results 