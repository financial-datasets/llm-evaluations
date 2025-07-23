from data import (
    RedFlagDetectionDatasetBuilder, 
    create_default_dataset,
    create_red_flags_only_dataset,
    FilterStrategy
)


def example_usage():
    """Demonstrate different ways to use the dataset builder."""
    
    # Example 1: Use the convenience function for default dataset
    print("=== Example 1: Default Dataset ===")
    dataset = create_default_dataset()
    print(f"Total companies: {dataset.size()}")
    print(f"Available labels: {dataset.labels()}")
    print(f"Red flag companies: {len(dataset.get_red_flag_companies())}")
    print(f"Green flag companies: {len(dataset.get_green_flag_companies())}")
    
    # Example 2: Configure specific filters with fluent interface
    print("\n=== Example 2: Custom Filter Configuration ===")
    custom_dataset = (RedFlagDetectionDatasetBuilder()
                     .with_financial_health_issues()
                     .with_bankruptcy_risk()
                     .with_green_flags()
                     .with_limit(10)  # Get more results per filter
                     .build())
    
    print(f"Custom dataset size: {custom_dataset.size()}")
    print(f"Available labels: {custom_dataset.labels()}")
    
    # Example 3: Get companies by specific criteria
    print("\n=== Example 3: Filter by Label ===")
    bankruptcy_companies = custom_dataset.get_companies_by_label("Bankruptcy Risk")
    print(f"Companies with bankruptcy risk: {len(bankruptcy_companies)}")
    for company in bankruptcy_companies[:3]:  # Show first 3
        print(f"  - {company['ticker']}: {company['label']}")
    
    # Example 4: Use convenience functions for specific datasets
    print("\n=== Example 4: Red Flags Only Dataset ===")
    red_flags_dataset = create_red_flags_only_dataset()
    print(f"Red flags only dataset: {red_flags_dataset.size()} companies")
    print(f"Labels: {red_flags_dataset.labels()}")
    
    # Example 5: Create a minimal dataset with just one filter
    print("\n=== Example 5: Single Filter Dataset ===")
    health_only_dataset = (RedFlagDetectionDatasetBuilder()
                          .with_financial_health_issues()
                          .with_limit(3)
                          .build())
    
    print(f"Health issues only dataset: {health_only_dataset.size()} companies")
    for company in health_only_dataset.get_companies():
        print(f"  - {company['ticker']}: {company['label']}")


# Example 6: Creating a custom filter strategy
class HighDebtFilter(FilterStrategy):
    """Custom filter for companies with extremely high debt."""
    
    def get_filters(self) -> list[dict[str, str]]:
        return [
            {"field": "total_debt", "operator": "gt", "value": 10_000_000_000},  # $10B+
            {"field": "debt_to_equity", "operator": "gt", "value": 3.0}
        ]
    
    def get_label(self) -> str:
        return "Extremely High Debt"


def custom_filter_example():
    """Demonstrate using a custom filter strategy."""
    print("\n=== Example 6: Custom Filter Strategy ===")
    
    custom_filter_dataset = (RedFlagDetectionDatasetBuilder()
                            .with_custom_filter(HighDebtFilter())
                            .with_green_flags()
                            .build())
    
    print(f"Dataset with custom filter: {custom_filter_dataset.size()} companies")
    high_debt_companies = custom_filter_dataset.get_companies_by_label("Extremely High Debt")
    print(f"Extremely high debt companies: {len(high_debt_companies)}")


if __name__ == "__main__":
    example_usage()
    custom_filter_example()
