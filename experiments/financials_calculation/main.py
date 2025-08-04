import os
from datetime import datetime
from data.factory import create_dataset
from experiments.financials_calculation.experiment import FinancialsCalculationExperiment
from experiments.financials_calculation.judge import FinancialsCalculationJudge

def main():
    """Load financials calculation dataset with JSON caching."""
    
    print("=== Financials Calculation Dataset ===")
    
    # Load dataset using the simplified factory
    dataset = create_dataset()
    
    # Display basic statistics
    print(f"Total companies: {dataset.size()}")

    # Run the experiment
    experiment = FinancialsCalculationExperiment()
    results = experiment.run(dataset)

    # # Evaluate the results
    judge = FinancialsCalculationJudge()
    evaluation_results = judge.evaluate(results)

    # # Pretty print the ComparisonResults
    print(evaluation_results.model_dump_json(indent=2))

    # Save the results to JSON files with timestamp
    current_dir = os.path.dirname(__file__)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save evaluation metrics (aggregated results)
    metrics_filepath = os.path.join(current_dir, f"financials_calculation_results_{timestamp}.json")
    with open(metrics_filepath, "w") as f:
        f.write(evaluation_results.model_dump_json(indent=2))
    print(f"📊 Evaluation metrics saved to: {metrics_filepath}")
    
    # Save raw predictions for manual validation
    predictions_filepath = os.path.join(current_dir, f"financials_calculation_predictions_{timestamp}.json")
    with open(predictions_filepath, "w") as f:
        f.write(results.model_dump_json(indent=2))
    print(f"🔍 Raw predictions saved to: {predictions_filepath}")

if __name__ == "__main__":
    main()
