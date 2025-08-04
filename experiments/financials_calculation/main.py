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

    # # Save the results to a JSON file with timestamp
    current_dir = os.path.dirname(__file__)
    json_filepath = os.path.join(current_dir, f"financials_calculation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_filepath, "w") as f:
        f.write(evaluation_results.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
