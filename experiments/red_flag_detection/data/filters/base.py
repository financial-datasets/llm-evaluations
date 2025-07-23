"""
Base filter strategy interface.

This module defines the abstract base class that all filter strategies must implement.
"""

from abc import ABC, abstractmethod


class FilterStrategy(ABC):
    """Abstract base class for filter strategies."""
    
    @abstractmethod
    def get_filters(self) -> list[dict[str, str]]:
        """Return the filter criteria for this strategy."""
        pass
    
    @abstractmethod
    def get_label(self) -> str:
        """Return the label for companies found with this filter."""
        pass 