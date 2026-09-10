import sys
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT SEMANTIC CLASSIFIER
# ============================================================

from semantic_intent_classifier import classify_intent


# ============================================================
# PATHS
# ============================================================

GOLDEN_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_set.csv"
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "semantic_predictions.csv"
)


# ============================================================
# LOAD GOLDEN SET
# ============================================================

print("=" * 70)
print("SEMANTIC INTENT CLASSIFIER EVALUATION")
print("=" * 70)

golden = pd.read_csv(
    GOLDEN_PATH
)

print(
    f"\nGolden examples: {len(golden)}"
)


# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\nRunning semantic predictions...")

predictions = []
confidences = []

for i, text in enumerate(
    golden["customer_text"],
    start=1,
):

    intent, confidence, _ = classify_intent(
        text
    )

    predictions.append(intent)
    confidences.append(confidence)

    if i % 25 == 0:
        print(
            f"Processed {i}/{len(golden)}"
        )


# ============================================================
# EVALUATION
# ============================================================

y_true = golden["intent"]
y_pred = predictions


accuracy = accuracy_score(
    y_true,
    y_pred,
)


macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SEMANTIC CLASSIFIER RESULTS")
print("=" * 70)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Macro-F1 : {macro_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

golden["predicted_intent"] = predictions
golden["confidence"] = confidences

golden.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\nPredictions saved to:")
print(OUTPUT_PATH)