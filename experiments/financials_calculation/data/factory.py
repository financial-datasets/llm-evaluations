import json
import os
import time
import random

from experiments.financials_calculation.data.dataset import FinancialsCalculationDataset

def create_dataset() -> FinancialsCalculationDataset:
    """Create a financials calculation dataset.
    
    Returns:
        FinancialsCalculationDataset containing companies with XBRL financial data
    """
    # Load the dataset.json file
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, "dataset.json"), "r") as f:
        dataset = json.load(f)
        return FinancialsCalculationDataset(dataset)
