import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "data/tourism.csv"
OUT = "model_building"

df = pd.read_csv(INPUT)
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Basic cleaning of inconsistent categorical labels
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["Occupation"] = df["Occupation"].replace({"Free Lancer": "Freelancer"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

# CustomerID is an identifier and is not used as a predictive feature.
X = df.drop(columns=["ProdTaken", "CustomerID"])
y = df["ProdTaken"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

os.makedirs(OUT, exist_ok=True)
X_train.to_csv(f"{OUT}/X_train.csv", index=False)
X_test.to_csv(f"{OUT}/X_test.csv", index=False)
y_train.to_csv(f"{OUT}/y_train.csv", index=False)
y_test.to_csv(f"{OUT}/y_test.csv", index=False)

print(f"Prepared {len(df)} rows; train={len(X_train)}, test={len(X_test)}")
