"""
Dataset builder for red flag detection.

This module provides the RedFlagDetectionDatasetBuilder class for constructing
datasets with configurable filter strategies.
"""

from typing import Optional
from clients.fd_client import FinancialDatasetsClient
from .dataset import RedFlagDetectionDataset
from .filters import (
    FilterStrategy,
    FinancialHealthIssuesFilter,
    DecliningProfitabilityFilter,
    EarningsDeclineFilter,
    BankruptcyRiskFilter,
    InefficientOperationsFilter,
    GreenFlagFilter,
)


class RedFlagDetectionDatasetBuilder:
    """Builder for constructing RedFlagDetectionDataset with configurable filters."""
    
    def __init__(self, fd_client: Optional[FinancialDatasetsClient] = None):
        self._fd_client = fd_client or FinancialDatasetsClient()
        self._filter_strategies: list[tuple[FilterStrategy, int]] = []
        self._default_limit = 5
        self._period = "ttm"
    
    def with_financial_health_issues(self, limit: Optional[int] = None) -> 'RedFlagDetectionDatasetBuilder':
        """Add financial health issues filter."""
        actual_limit = limit if limit is not None else self._default_limit
        self._filter_strategies.append((FinancialHealthIssuesFilter(), actual_limit))
        return self
    
    def with_declining_profitability(self, limit: Optional[int] = None) -> 'RedFlagDetectionDatasetBuilder':
        """Add declining profitability filter."""
        actual_limit = limit if limit is not None else self._default_limit
        self._filter_strategies.append((DecliningProfitabilityFilter(), actual_limit))
        return self
    
    def with_earnings_decline(self, limit: Optional[int] = None) -> 'RedFlagDetectionDatasetBuilder':
        """Add earnings decline filter."""
        actual_limit = limit if limit is not None else self._default_limit
        self._filter_strategies.append((EarningsDeclineFilter(), actual_limit))
        return self
    
    def with_bankruptcy_risk(self, limit: Optional[int] = None) -> 'RedFlagDetectionDatasetBuilder':
        """Add bankruptcy risk filter."""
        actual_limit = limit if limit is not None else self._default_limit
        self._filter_strategies.append((BankruptcyRiskFilter(), actual_limit))
        return self
    
    def with_inefficient_operations(self, limit: Optional[int] = None) -> 'RedFlagDetectionDatasetBuilder':
        """Add inefficient operations filter."""
        actual_limit = limit if limit is not None else self._default_limit
        self._filter_strategies.append((InefficientOperationsFilter(), actual_limit))
        return self
    
    def with_green_flags(self, limit: Optional[int] = None) -> 'RedFlagDetectionDatasetBuilder':
        """Add green flag filter."""
        actual_limit = limit if limit is not None else self._default_limit
        self._filter_strategies.append((GreenFlagFilter(), actual_limit))
        return self
    
    def with_all_red_flags(self, limit_per_filter: Optional[int] = None) -> 'RedFlagDetectionDatasetBuilder':
        """Add all red flag filters with optional limit per filter."""
        return (self.with_financial_health_issues(limit_per_filter)
                   .with_declining_profitability(limit_per_filter)
                   .with_earnings_decline(limit_per_filter)
                   .with_bankruptcy_risk(limit_per_filter)
                   .with_inefficient_operations(limit_per_filter))
    
    def with_custom_filter(self, filter_strategy: FilterStrategy, limit: Optional[int] = None) -> 'RedFlagDetectionDatasetBuilder':
        """Add a custom filter strategy."""
        actual_limit = limit if limit is not None else self._default_limit
        self._filter_strategies.append((filter_strategy, actual_limit))
        return self
    
    def with_limit(self, limit: int) -> 'RedFlagDetectionDatasetBuilder':
        """Set the default limit for filters that don't specify their own limit."""
        self._default_limit = limit
        return self
    
    def with_period(self, period: str) -> 'RedFlagDetectionDatasetBuilder':
        """Set the period for financial data (e.g., 'ttm', 'annual')."""
        self._period = period
        return self
    
    def build(self) -> RedFlagDetectionDataset:
        """Build and return the dataset."""
        if not self._filter_strategies:
            raise ValueError("At least one filter strategy must be added before building the dataset")
        
        all_companies = []
        for strategy, limit in self._filter_strategies:
            companies = self._fd_client.search(
                filters=strategy.get_filters(),
                label=strategy.get_label(),
                limit=limit,
                period=self._period
            )
            print(f"Found {len(companies)} companies with label {strategy.get_label()} (limit: {limit})")
            all_companies.extend(companies)
        
        return RedFlagDetectionDataset(all_companies) 