"""
Caching utilities for red flag detection datasets.

This module provides convenience functions that handle JSON caching
for commonly used dataset configurations.
"""

import os
from .dataset import RedFlagDetectionDataset
from .builder import RedFlagDetectionDatasetBuilder


def create_default_dataset(json_filepath: str = None) -> RedFlagDetectionDataset:
    """Create a dataset with all red flags and green flags using default settings.
    
    Args:
        json_filepath: Path to save/load the JSON cache file
    """
    if json_filepath is None:
        # Save in the data module directory
        data_dir = os.path.dirname(__file__)
        json_filepath = os.path.join(data_dir, "default_dataset.json")
    
    # Try to load from JSON first
    dataset = RedFlagDetectionDataset.load_from_json(json_filepath)
    if dataset is not None:
        print(f"Loaded cached dataset from {json_filepath}")
        return dataset
    
    # If no cached data, build from scratch
    print("No cached dataset found. Building from API...")
    dataset = (RedFlagDetectionDatasetBuilder()
               .with_all_red_flags()
               .with_green_flags()
               .build())
    
    # Save to JSON for next time
    dataset.save_to_json(json_filepath)
    return dataset


def create_custom_limit_dataset(json_filepath: str = None) -> RedFlagDetectionDataset:
    """Create a dataset with custom limits per filter.
    
    Example: 10 total red flag companies (2 each from 5 red flag filters) 
    and 10 green flag companies.
    
    Args:
        json_filepath: Path to save/load the JSON cache file
    """
    if json_filepath is None:
        # Save in the data module directory
        data_dir = os.path.dirname(__file__)
        json_filepath = os.path.join(data_dir, "custom_limit_dataset.json")
    
    # Try to load from JSON first
    dataset = RedFlagDetectionDataset.load_from_json(json_filepath)
    if dataset is not None:
        print(f"Loaded cached dataset from {json_filepath}")
        return dataset
    
    # If no cached data, build from scratch
    print("No cached dataset found. Building from API...")
    dataset = (RedFlagDetectionDatasetBuilder()
               .with_financial_health_issues(limit=2)
               .with_declining_profitability(limit=2)
               .with_earnings_decline(limit=2)
               .with_bankruptcy_risk(limit=2)
               .with_inefficient_operations(limit=2)
               .with_green_flags(limit=10)
               .build())
    
    # Save to JSON for next time
    dataset.save_to_json(json_filepath)
    return dataset


def create_red_flags_only_dataset(json_filepath: str = None) -> RedFlagDetectionDataset:
    """Create a dataset with only red flag companies.
    
    Args:
        json_filepath: Path to save/load the JSON cache file
    """
    if json_filepath is None:
        # Save in the data module directory
        data_dir = os.path.dirname(__file__)
        json_filepath = os.path.join(data_dir, "red_flags_dataset.json")
    
    # Try to load from JSON first
    dataset = RedFlagDetectionDataset.load_from_json(json_filepath)
    if dataset is not None:
        return dataset
    
    # If no cached data, build from scratch
    print("No cached dataset found. Building from API...")
    dataset = (RedFlagDetectionDatasetBuilder()
               .with_all_red_flags()
               .build())
    
    # Save to JSON for next time
    dataset.save_to_json(json_filepath)
    return dataset


def create_green_flags_only_dataset(json_filepath: str = None) -> RedFlagDetectionDataset:
    """Create a dataset with only green flag companies.
    
    Args:
        json_filepath: Path to save/load the JSON cache file
    """
    if json_filepath is None:
        # Save in the data module directory
        data_dir = os.path.dirname(__file__)
        json_filepath = os.path.join(data_dir, "green_flags_dataset.json")
    
    # Try to load from JSON first
    dataset = RedFlagDetectionDataset.load_from_json(json_filepath)
    if dataset is not None:
        return dataset
    
    # If no cached data, build from scratch
    print("No cached dataset found. Building from API...")
    dataset = (RedFlagDetectionDatasetBuilder()
               .with_green_flags()
               .build())
    
    # Save to JSON for next time
    dataset.save_to_json(json_filepath)
    return dataset 