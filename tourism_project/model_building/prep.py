import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")   # the registered tourism.csv inside the data folder
print(f"Raw shape: {df.shape}")

# ---------------------------------------------------------------- cleaning
# 1) Drop the leftover index column and the customer identifier: neither is a predictive feature
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

# 2) Fix the typo category in Gender ("Fe Male" -> "Female")
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# 3) Remove exact duplicate customers (same values in every column). Duplicates can land in
#    both train and test, which would make the evaluation look better than it really is.
n_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicate rows removed: {n_before - len(df)}  ->  clean shape: {df.shape}")

# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them, and the Streamlit app also sends
# raw category values. Encoding them here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

target = "ProdTaken"  # 1 if the customer purchased the package, else 0
X = df.drop(columns=[target])
y = df[target]

# stratify keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   # stratify on the target
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print(f"Train: {Xtrain.shape}  Test: {Xtest.shape}")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
print("ProdTaken distribution in test:")
print(ytest.value_counts())
