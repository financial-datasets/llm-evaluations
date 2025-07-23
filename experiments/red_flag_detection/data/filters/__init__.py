"""
Filter strategies for red flag detection.

This package contains all filter implementations for identifying
companies with various financial characteristics.
"""

from .base import FilterStrategy
from .red_flag_filters import (
    FinancialHealthIssuesFilter,
    DecliningProfitabilityFilter,
    EarningsDeclineFilter,
    BankruptcyRiskFilter,
    InefficientOperationsFilter,
)
from .green_flag_filters import GreenFlagFilter

__all__ = [
    "FilterStrategy",
    "FinancialHealthIssuesFilter", 
    "DecliningProfitabilityFilter",
    "EarningsDeclineFilter",
    "BankruptcyRiskFilter",
    "InefficientOperationsFilter",
    "GreenFlagFilter",
] 