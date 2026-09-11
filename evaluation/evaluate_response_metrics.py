import os
import sys
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "golden",
    "human_response_evaluation.csv",
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AUTOMATED RESPONSE METRICS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(
    f"\nExamples evaluated: {len(df)}"
)


# ============================================================
# BASIC RESPONSE METRICS
# ============================================================

df["response_word_count"] = (
    df["generated_response"]
    .fillna("")
    .astype(str)
    .str.split()
    .str.len()
)


df["response_char_count"] = (
    df["generated_response"]
    .fillna("")
    .astype(str)
    .str.len()
)


# ============================================================
# HUMAN SCORE METRICS
# ============================================================

score_columns = [
    "relevance_score",
    "groundedness_score",
    "helpfulness_score",
    "overall_score",
]


for column in score_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("HUMAN RESPONSE QUALITY")
print("=" * 70)

for column in score_columns:

    print(
        f"{column:25} "
        f"{df[column].mean():.2f} / 5"
    )


# ============================================================
# INTENT PERFORMANCE
# ============================================================

intent_accuracy = (
    df["intent_correct"]
    .astype(bool)
    .mean()
    * 100
)

print("\n" + "=" * 70)
print("INTENT PERFORMANCE")
print("=" * 70)

print(
    f"\nIntent accuracy: "
    f"{intent_accuracy:.2f}%"
)


# ============================================================
# RETRIEVAL QUALITY
# ============================================================

retrieval_scores = pd.to_numeric(
    df["top_similarity"],
    errors="coerce",
).dropna()


print("\n" + "=" * 70)
print("RETRIEVAL")
print("=" * 70)

if len(retrieval_scores) > 0:

    print(
        f"\nAverage top similarity: "
        f"{retrieval_scores.mean():.4f}"
    )

    print(
        f"Median top similarity : "
        f"{retrieval_scores.median():.4f}"
    )

    print(
        f"Minimum similarity    : "
        f"{retrieval_scores.min():.4f}"
    )

    print(
        f"Maximum similarity    : "
        f"{retrieval_scores.max():.4f}"
    )


# ============================================================
# RESPONSE LENGTH
# ============================================================

print("\n" + "=" * 70)
print("RESPONSE LENGTH")
print("=" * 70)

print(
    f"\nAverage words: "
    f"{df['response_word_count'].mean():.2f}"
)

print(
    f"Median words : "
    f"{df['response_word_count'].median():.2f}"
)


# ============================================================
# DECISION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ESCALATION DECISIONS")
print("=" * 70)

print(
    "\n"
    + df["decision"]
    .value_counts()
    .to_string()
)


# ============================================================
# AUTO-HANDLE QUALITY
# ============================================================

auto_handle = df[
    df["decision"]
    == "AUTO_HANDLE"
]

if len(auto_handle) > 0:

    print("\n" + "=" * 70)
    print("AUTO-HANDLE QUALITY")
    print("=" * 70)

    print(
        f"\nAuto-handled examples: "
        f"{len(auto_handle)}"
    )

    print(
        f"Average overall human score: "
        f"{auto_handle['overall_score'].mean():.2f} / 5"
    )

    print(
        f"Intent accuracy: "
        f"{auto_handle['intent_correct'].astype(bool).mean() * 100:.2f}%"
    )


# ============================================================
# ESCALATED QUALITY
# ============================================================

escalated = df[
    df["decision"]
    == "ESCALATE_TO_HUMAN"
]

if len(escalated) > 0:

    print("\n" + "=" * 70)
    print("ESCALATED CASE QUALITY")
    print("=" * 70)

    print(
        f"\nEscalated examples: "
        f"{len(escalated)}"
    )

    print(
        f"Average overall human score: "
        f"{escalated['overall_score'].mean():.2f} / 5"
    )


# ============================================================
# LEAKAGE CHECK
# ============================================================

leakage = df[
    df["leakage_detected"]
    .astype(bool)
]

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

print(
    f"\nLeakage cases: "
    f"{len(leakage)}"
)


# ============================================================
# SAVE METRICS DATA
# ============================================================

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "golden",
    "response_metrics.csv",
)

df.to_csv(
    OUTPUT_PATH,
    index=False,
)


print(
    f"\nDetailed metrics saved to:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# DONE
# ============================================================

print("\nDone!")