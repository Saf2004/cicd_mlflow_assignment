import os
import pandas as pd
from sklearn.datasets import make_classification

os.makedirs("data", exist_ok=True)

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_classes=3,
    n_informative=5,
    n_redundant=2,
    random_state=42,
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
df["label"] = y
df.to_csv("data/train.csv", index=False)
print(f"Generated data/train.csv with {len(df)} rows and {df['label'].nunique()} classes")
