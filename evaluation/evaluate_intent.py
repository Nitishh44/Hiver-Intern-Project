import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR),
)


# ============================================================
# IMPORT CLASSIFIER
# ============================================================

from semantic_intent_classifier import (
    classify_intent,
)


# ============================================================
# DATA PATH
# ============================================================

GOLDEN_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_set.csv"
)


# ============================================================
# LOAD GOLDEN SET
# ============================================================

golden = pd.read_csv(
    GOLDEN_PATH
)

print("=" * 70)
print("SEMANTIC INTENT CLASSIFIER EVALUATION")
print("=" * 70)

print(
    f"\nGolden examples: {len(golden)}"
)

print(
    "Golden set loaded successfully."
)


# ============================================================
# PREDICTIONS
# ============================================================

predictions = []
confidences = []
second_intents = []
second_scores = []
intent_margins = []


print("\nRunning semantic predictions...")


for i, text in enumerate(
    golden["customer_text"],
    start=1,
):

    (
        intent,
        confidence,
        ranked_intents,
        second_intent,
        second_score,
        intent_margin,
    ) = classify_intent(
        text
    )

    predictions.append(
        intent
    )

    confidences.append(
        confidence
    )

    second_intents.append(
        second_intent
    )

    second_scores.append(
        second_score
    )

    intent_margins.append(
        intent_margin
    )

    if i % 25 == 0:
        print(
            f"Processed {i}/{len(golden)}"
        )


# ============================================================
# ADD RESULTS
# ============================================================

golden["predicted_intent"] = predictions

golden["confidence"] = confidences

golden["second_intent"] = second_intents

golden["second_score"] = second_scores

golden["intent_margin"] = intent_margins


# ============================================================
# METRICS
# ============================================================

y_true = golden["intent"]

y_pred = golden["predicted_intent"]


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
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    f"\nAccuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Macro-F1  : {macro_f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )
)


# ============================================================
# CONFIDENCE / MARGIN SUMMARY
# ============================================================

print("=" * 70)
print("CONFIDENCE / MARGIN SUMMARY")
print("=" * 70)

print(
    f"\nAverage confidence : "
    f"{golden['confidence'].mean():.4f}"
)

print(
    f"Average margin     : "
    f"{golden['intent_margin'].mean():.4f}"
)

print(
    f"Minimum margin     : "
    f"{golden['intent_margin'].min():.4f}"
)

print(
    f"Maximum margin     : "
    f"{golden['intent_margin'].max():.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "intent_evaluation.csv"
)

golden.to_csv(
    OUTPUT_PATH,
    index=False,
)


print("\nEvaluation saved to:")

print(
    OUTPUT_PATH
)

print("\nDone.")