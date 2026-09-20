import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"  # raw file registered inside the repo's data folder

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# Extra columns are reported (not fatal): the raw export carries a leftover index column
extra = [c for c in df.columns if c not in expected_columns]

# Basic integrity checks on the key columns
if df["CustomerID"].duplicated().any():
    raise ValueError("CustomerID must be unique.")
if not set(df["ProdTaken"].unique()) <= {0, 1}:
    raise ValueError("ProdTaken must be binary (0/1).")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Extra (non-dictionary) columns to be dropped in data prep:", extra)
print("Missing values in total:", int(df.isna().sum().sum()))
print("ProdTaken distribution:")
print(df["ProdTaken"].value_counts())
