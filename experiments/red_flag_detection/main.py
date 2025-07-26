from data.factory import create_dataset
from experiments.red_flag_detection.experiment import RedFlagDetectionExperiment
from experiments.red_flag_detection.judge import RedFlagDetectionJudge


def main():
    """Load red flag and green flag companies dataset with JSON caching."""
    
    print("=== Red Flag Detection Dataset ===")
    
    # Load dataset using the simplified factory
    dataset = create_dataset()
    
    # Display basic statistics
    print(f"Total companies: {dataset.size()}")
    print(f"Red flag companies: {len(dataset.get_red_flag_companies())}")
    print(f"Green flag companies: {len(dataset.get_green_flag_companies())}")

    # Run the experiment
    experiment = RedFlagDetectionExperiment()
    results = experiment.run(dataset)

    # Evaluate the results
    judge = RedFlagDetectionJudge()
    evaluation_results = judge.evaluate(results)

    # Pretty print the ComparisonResults
    print(evaluation_results.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
