"""
Red Flag Detection Package

This package provides tools for building datasets to detect financial red flags
in companies using various filter strategies and a flexible builder pattern.

Main Components:
- RedFlagDetectionDataset: Container for financial data with classification methods
- RedFlagDetectionDatasetBuilder: Configurable builder for constructing datasets
- FilterStrategy: Base class for implementing custom filters
- Various pre-built filter strategies for common red/green flag scenarios

Example Usage:
    # Quick start with default configuration
    from experiments.red_flag_detection import create_default_dataset
    dataset = create_default_dataset()
    
    # Custom configuration
    from experiments.red_flag_detection import RedFlagDetectionDatasetBuilder
    dataset = (RedFlagDetectionDatasetBuilder()
               .with_financial_health_issues()
               .with_bankruptcy_risk()
               .with_limit(10)
               .build())
"""

import sys
import os

# Add project root to path to enable absolute imports from this package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from .dataset import RedFlagDetectionDataset
from .builder import RedFlagDetectionDatasetBuilder
from .filters import (
    FilterStrategy,
    FinancialHealthIssuesFilter,
    DecliningProfitabilityFilter,
    EarningsDeclineFilter,
    BankruptcyRiskFilter,
    InefficientOperationsFilter,
    GreenFlagFilter,
)


def create_default_dataset() -> RedFlagDetectionDataset:
    """Create a dataset with all red flags and green flags using default settings."""
    return (RedFlagDetectionDatasetBuilder()
            .with_all_red_flags()
            .with_green_flags()
            .build())


def create_red_flags_only_dataset() -> RedFlagDetectionDataset:
    """Create a dataset with only red flag companies."""
    return (RedFlagDetectionDatasetBuilder()
            .with_all_red_flags()
            .build())


def create_green_flags_only_dataset() -> RedFlagDetectionDataset:
    """Create a dataset with only green flag companies."""
    return (RedFlagDetectionDatasetBuilder()
            .with_green_flags()
            .build())


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