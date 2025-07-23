from data.cache import create_default_dataset, create_custom_limit_dataset


def main():
    """Load red flag and green flag companies dataset with JSON caching."""
    
    print("=== Red Flag Detection Dataset with Custom Limits ===")
    
    # Load dataset with custom limits per filter
    # This will create 10 total red flag companies (2 each from 5 red flag filters)
    # and 10 green flag companies
    dataset = create_custom_limit_dataset()
    
    # Display basic statistics
    print(f"Total companies: {dataset.size()}")
    print(f"Available labels: {dataset.labels()}")
    print(f"Red flag companies: {len(dataset.get_red_flag_companies())}")
    print(f"Green flag companies: {len(dataset.get_green_flag_companies())}")
    
    # Show breakdown by label
    print("\n=== Companies by Label ===")
    for label in sorted(dataset.labels()):
        companies = dataset.get_companies_by_label(label)
        print(f"{label}: {len(companies)} companies")
        
        # Show a few example companies for each label
        for company in companies[:3]:  # Show first 3 companies
            print(f"  - {company['ticker']}")
    
    # Separate red flag and green flag companies
    print("\n=== Red Flag Companies ===")
    red_flag_companies = dataset.get_red_flag_companies()
    print(f"Found {len(red_flag_companies)} red flag companies")
    for company in red_flag_companies[:5]:  # Show first 5
        print(f"  - {company['ticker']}: {company['label']}")
    
    print("\n=== Green Flag Companies ===")
    green_flag_companies = dataset.get_green_flag_companies()
    print(f"Found {len(green_flag_companies)} green flag companies")
    for company in green_flag_companies[:5]:  # Show first 5
        print(f"  - {company['ticker']}: {company['label']}")


if __name__ == "__main__":
    main()
