import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = "data/golden/golden_set.csv"

df = pd.read_csv(DATA_PATH)

# Majority class from the golden set
majority_class = df["intent"].value_counts().idxmax()

# Predict the same class for every example
y_true = df["intent"]
y_pred = [majority_class] * len(df)

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

print("=" * 70)
print("BASELINE #1 — MAJORITY CLASS")
print("=" * 70)

print(f"Examples       : {len(df)}")
print(f"Majority class : {majority_class}")
print(f"Accuracy       : {accuracy:.4f}")
print(f"Macro F1       : {macro_f1:.4f}")

print("\nPercentage:")
print(f"Accuracy       : {accuracy:.2%}")
print(f"Macro F1       : {macro_f1:.2%}")