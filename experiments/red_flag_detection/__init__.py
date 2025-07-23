"""
Red Flag Detection Experiment

This experiment provides tools for building datasets to detect financial red flags
in companies using various filter strategies and a flexible builder pattern.
"""

from .data import (
    RedFlagDetectionDataset,
    RedFlagDetectionDatasetBuilder,
    FilterStrategy,
    FinancialHealthIssuesFilter,
    DecliningProfitabilityFilter,
    EarningsDeclineFilter,
    BankruptcyRiskFilter,
    InefficientOperationsFilter,
    GreenFlagFilter,
    create_default_dataset,
    create_red_flags_only_dataset,
    create_green_flags_only_dataset,
)

__all__ = [
    # Main classes
    "RedFlagDetectionDataset",
    "RedFlagDetectionDatasetBuilder",
    
    # Filter base class
    "FilterStrategy",
    
    # Pre-built filters
    "FinancialHealthIssuesFilter",
    "DecliningProfitabilityFilter", 
    "EarningsDeclineFilter",
    "BankruptcyRiskFilter",
    "InefficientOperationsFilter",
    "GreenFlagFilter",
    
    # Convenience functions
    "create_default_dataset",
    "create_red_flags_only_dataset", 
    "create_green_flags_only_dataset",
] 