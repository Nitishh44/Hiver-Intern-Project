import os
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# ============================================================
# PATHS
# ============================================================

RESPONSE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "golden",
    "response_evaluation.csv",
)

HUMAN_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "golden",
    "human_response_evaluation.csv",
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "golden",
    "escalation_threshold_evaluation.csv",
)


# ============================================================
# SETTINGS
# ============================================================

MARGIN_THRESHOLDS = [
    0.00,
    0.02,
    0.03,
    0.05,
    0.08,
    0.10,
]

HIGH_RISK_INTENTS = {
    "billing_payment",
    "account_access",
    "order_purchase",
}


# ============================================================
# LOAD + MERGE DATA
# ============================================================

print("=" * 70)
print("ESCALATION THRESHOLD EVALUATION")
print("=" * 70)

response_df = pd.read_csv(RESPONSE_PATH)
human_df = pd.read_csv(HUMAN_PATH)

print(
    f"\nResponse evaluation rows: {len(response_df)}"
)

print(
    f"Response rating rows:     {len(human_df)}"
)


# Keep only required human-rating columns
human_df = human_df[
    [
        "golden_id",
        "overall_score",
    ]
]


# Merge model outputs with response-quality ratings
df = response_df.merge(
    human_df,
    on="golden_id",
    how="inner",
)


print(
    f"Merged evaluation rows:   {len(df)}"
)


# ============================================================
# CURRENT POLICY
# ============================================================

def current_policy(row):

    intent = row["predicted_intent"]
    confidence = float(
        row["intent_confidence"]
    )

    similarity = row["top_similarity"]

    if intent in HIGH_RISK_INTENTS:
        return "ESCALATE_TO_HUMAN"

    if pd.isna(similarity):
        return "ESCALATE_TO_HUMAN"

    if confidence < 0.30:
        return "ESCALATE_TO_HUMAN"

    if float(similarity) >= 0.75:
        return "AUTO_HANDLE"

    return "ESCALATE_TO_HUMAN"


# ============================================================
# MARGIN POLICY
# ============================================================

def margin_policy(row, margin_threshold):

    intent = row["predicted_intent"]

    confidence = float(
        row["intent_confidence"]
    )

    similarity = row["top_similarity"]

    margin = float(
        row["intent_margin"]
    )


    # High-risk intents
    if intent in HIGH_RISK_INTENTS:
        return "ESCALATE_TO_HUMAN"


    # No retrieval evidence
    if pd.isna(similarity):
        return "ESCALATE_TO_HUMAN"

    similarity = float(similarity)


    # Low confidence
    if confidence < 0.30:
        return "ESCALATE_TO_HUMAN"


    # Weak retrieval evidence
    if similarity < 0.75:
        return "ESCALATE_TO_HUMAN"


    # Ambiguous intent
    if margin < margin_threshold:
        return "ESCALATE_TO_HUMAN"


    return "AUTO_HANDLE"


# ============================================================
# ADD CURRENT POLICY
# ============================================================

df["current_policy"] = df.apply(
    current_policy,
    axis=1,
)


# ============================================================
# THRESHOLD SWEEP
# ============================================================

results = []


print("\n" + "=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)


for threshold in MARGIN_THRESHOLDS:

    policy_column = (
        f"margin_{threshold:.2f}"
    )


    df[policy_column] = df.apply(
        lambda row: margin_policy(
            row,
            threshold,
        ),
        axis=1,
    )


    auto = df[
        df[policy_column]
        == "AUTO_HANDLE"
    ]


    escalated = df[
        df[policy_column]
        == "ESCALATE_TO_HUMAN"
    ]


    auto_count = len(auto)

    escalated_count = len(
        escalated
    )


    # --------------------------------------------------------
    # AUTO METRICS
    # --------------------------------------------------------

    if auto_count > 0:

        auto_accuracy = (
            auto["intent_correct"].mean()
            * 100
        )

        auto_quality = (
            auto["overall_score"].mean()
        )

        auto_similarity = (
            auto["top_similarity"].mean()
        )

    else:

        auto_accuracy = None
        auto_quality = None
        auto_similarity = None


    # --------------------------------------------------------
    # ESCALATED QUALITY
    # --------------------------------------------------------

    if escalated_count > 0:

        escalated_quality = (
            escalated["overall_score"].mean()
        )

    else:

        escalated_quality = None


    # --------------------------------------------------------
    # DECISION CHANGES
    # --------------------------------------------------------

    changed_decisions = (
        df[policy_column]
        != df["current_policy"]
    ).sum()


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append(
        {
            "margin_threshold": threshold,
            "auto_cases": auto_count,
            "escalated_cases": escalated_count,
            "auto_intent_accuracy": auto_accuracy,
            "auto_overall_score": auto_quality,
            "auto_avg_similarity": auto_similarity,
            "escalated_overall_score": escalated_quality,
            "changed_from_current": changed_decisions,
        }
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"\nMargin threshold >= {threshold:.2f}"
    )

    print(
        f"  Auto-handled         : "
        f"{auto_count}"
    )

    print(
        f"  Escalated            : "
        f"{escalated_count}"
    )

    if auto_accuracy is not None:

        print(
            f"  Auto intent accuracy : "
            f"{auto_accuracy:.2f}%"
        )

        print(
            f"  Auto overall score   : "
            f"{auto_quality:.2f} / 5"
        )

        print(
            f"  Auto avg similarity  : "
            f"{auto_similarity:.4f}"
        )

    else:

        print(
            "  Auto metrics         : N/A"
        )


    if escalated_quality is not None:

        print(
            f"  Escalated quality    : "
            f"{escalated_quality:.2f} / 5"
        )

    else:

        print(
            "  Escalated quality    : N/A"
        )


    print(
        f"  Changed decisions    : "
        f"{changed_decisions}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD SUMMARY")
print("=" * 70)

print()

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(
    f"\nSaved to:\n{OUTPUT_PATH}"
)

print("\nDone!")