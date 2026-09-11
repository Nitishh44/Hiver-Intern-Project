import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "golden",
    "response_evaluation.csv",
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "golden",
    "human_response_evaluation.csv",
)


# ============================================================
# LOAD RESPONSE EVALUATION
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("HUMAN RESPONSE EVALUATION")
print("=" * 70)

print(
    f"\nExamples available: {len(df)}"
)


# ============================================================
# CREATE HUMAN RATING COLUMNS
# ============================================================

df["relevance_score"] = " "
df["groundedness_score"] = " "
df["helpfulness_score"] = " "
df["overall_score"] = " "
df["human_notes"] = " "


# ============================================================
# SAVE ANNOTATION TEMPLATE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False,
)


print(
    "\nHuman evaluation template created:"
)

print(
    OUTPUT_PATH
)

print("\nRating scale:")

print("""
1 = Poor
2 = Weak
3 = Acceptable
4 = Good
5 = Excellent

Rate each generated response on:

Relevance:
Does the response address the customer's actual problem?

Groundedness:
Is the response consistent with the retrieved historical
support evidence and does it avoid unsupported claims?

Helpfulness:
Would this response reasonably help the customer move
toward resolving the issue?

Overall:
Your overall judgment of response quality.
""")

print("\nDone!")