from pathlib import Path

DATASET_SLUG = "mohankrishnathalla/global-house-purchase-decision-dataset"
DATA_FILENAME = "global_house_purchase_dataset.csv"
DEFAULT_DATA = Path("data/raw") / DATA_FILENAME
SEED = 42
TARGET = "decision"
IDENTIFIER = "property_id"
CATEGORICAL = ["country", "city", "property_type", "furnishing_status"]
NUMERICAL = [
    "property_size_sqft", "price", "constructed_year", "previous_owners",
    "rooms", "bathrooms", "garage", "garden", "crime_cases_reported",
    "legal_cases_on_property", "customer_salary", "loan_amount",
    "loan_tenure_years", "monthly_expenses", "down_payment",
    "emi_to_income_ratio", "satisfaction_score", "neighbourhood_rating",
    "connectivity_score",
]
FEATURES = NUMERICAL + CATEGORICAL
