import os
from clients.fd_client import FinancialDatasetsClient
from experiments.red_flag_detection.data.dataset import RedFlagDetectionDataset

def get_red_flag_companies(fd_client: FinancialDatasetsClient) -> list[dict[str, str]]:
    """Get red flag companies from the FinancialDatasetsClient."""
    # Create red flag filters and get companies
    red_flag_configs = [
        {
            "filters": [
                {"field": "current_ratio", "operator": "lt", "value": 1.0},
                {"field": "quick_ratio", "operator": "lt", "value": 0.8},
                {"field": "debt_to_equity", "operator": "gt", "value": 2.0},
                {"field": "total_debt", "operator": "gt", "value": 2000000000}
            ],
            "label": "Financial Health Issues",
            "limit": 4
        },
        {
            "filters": [
                {"field": "net_margin", "operator": "lt", "value": 5.0},
                {"field": "operating_margin", "operator": "lt", "value": 5.0},
                {"field": "net_income", "operator": "lt", "value": 0}
            ],
            "label": "Declining Profitability",
            "limit": 2
        },
        {
            "filters": [
                {"field": "earnings_growth", "operator": "lt", "value": 0},
                {"field": "free_cash_flow_growth", "operator": "lt", "value": 0},
                {"field": "revenue_growth", "operator": "lt", "value": 0}
            ],
            "label": "Earnings Decline",
            "limit": 2
        },
        {
            "filters": [
                {"field": "inventory_turnover", "operator": "lt", "value": 2.0},
                {"field": "receivables_turnover", "operator": "lt", "value": 4.0},
                {"field": "asset_turnover", "operator": "lt", "value": 0.5}
            ],
            "label": "Inefficient Operations",
            "limit": 2
        }
    ]
    
    # Fetch red flag companies
    red_flag_companies = []
    for config in red_flag_configs:
        companies = fd_client.search(
            filters=config["filters"],
            label=config["label"],
            limit=config["limit"],
            period="ttm"
        )
        print(f"Found {len(companies)} companies with label {config.get('label')}")
        red_flag_companies.extend(companies)
    return red_flag_companies

def get_green_flag_companies(fd_client: FinancialDatasetsClient) -> list[dict[str, str]]:
    """Get green flag companies from the FinancialDatasetsClient."""
    green_flag_filters = [
        {"field": "net_income", "operator": "gte", "value": 250000000},
        {"field": "total_debt", "operator": "lt", "value": 2000000000},
        {"field": "current_ratio", "operator": "gte", "value": 1.2},
        {"field": "free_cash_flow", "operator": "gte", "value": 100000000}
    ]
    
    green_flag_companies = fd_client.search(
        filters=green_flag_filters,
        label="Green Flag",
        limit=10,
        period="ttm"
    )
    print(f"Found {len(green_flag_companies)} companies with label Green Flag")
    return green_flag_companies

def create_dataset() -> RedFlagDetectionDataset:
    """Create a red flag detection dataset with both red and green flag companies.
    
    Args:
        json_filepath: Optional path to save/load the dataset as JSON
        
    Returns:
        RedFlagDetectionDataset containing companies with various flag labels
    """
    # Check if the dataset already exists at the json_filepath
    data_dir = os.path.dirname(__file__)
    json_filepath = os.path.join(data_dir, "dataset.json")
    existing_dataset = RedFlagDetectionDataset.load_from_json(json_filepath)
    if existing_dataset is not None:
        print(f"Loaded existing dataset from {json_filepath}")
        return existing_dataset
    
    # If the dataset does not exist, create it
    print("No cached dataset found. Building from API...")
    fd_client = FinancialDatasetsClient()
    all_companies = []

    # Get red flag companies
    red_flag_companies = get_red_flag_companies(fd_client)
    all_companies.extend(red_flag_companies)

    # Get green flag companies
    green_flag_companies = get_green_flag_companies(fd_client)
    all_companies.extend(green_flag_companies)
    
    # Combine the datasets and return
    dataset = RedFlagDetectionDataset(all_companies)
    
    # Save to JSON for future use
    dataset.save_to_json(json_filepath)
    
    return dataset