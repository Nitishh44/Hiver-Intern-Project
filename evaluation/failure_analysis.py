import pandas as pd


def main():
    response_path = "data/golden/response_evaluation.csv"
    human_path = "data/golden/human_response_evaluation.csv"

    response = pd.read_csv(response_path)
    human = pd.read_csv(human_path)

    # Bring manual/reference quality ratings into evaluation data
    ratings = human[
        [
            "golden_id",
            "relevance_score",
            "groundedness_score",
            "helpfulness_score",
            "overall_score",
            "human_notes",
        ]
    ]

    df = pd.merge(response, ratings, on="golden_id", how="inner")

    print("Total matched examples:", len(df))

    # ---------------------------------------------------------
    # 1. Wrong intent predictions
    # ---------------------------------------------------------
    failures = df[
        df["expected_intent"] != df["predicted_intent"]
    ].copy()

    print("\nPotential failure cases:")
    print("=" * 70)

    print(f"\nWrong intent predictions: {len(failures)}")

    for _, row in failures.iterrows():
        print("\nGolden ID:", row["golden_id"])
        print("Customer:", row["customer_text"])
        print("Expected:", row["expected_intent"])
        print("Predicted:", row["predicted_intent"])
        print("Confidence:", round(row["intent_confidence"], 4))
        print("Second intent:", row["second_intent"])
        print("Margin:", round(row["intent_margin"], 4))
        print("Decision:", row["decision"])
        print("Similarity:", round(row["top_similarity"], 4))
        print("Helpfulness:", row["helpfulness_score"])
        print("Overall:", row["overall_score"])
        print("Response:", row["generated_response"])
        print("-" * 70)

    # ---------------------------------------------------------
    # 2. Low-helpfulness responses
    # ---------------------------------------------------------
    print("\n\nLow-helpfulness cases")
    print("=" * 70)

    low_help = df[
        df["helpfulness_score"] <= 2
    ].copy()

    print(f"\nLow-helpfulness responses: {len(low_help)}")

    for _, row in low_help.iterrows():
        print("\nGolden ID:", row["golden_id"])
        print("Customer:", row["customer_text"])
        print("Expected:", row["expected_intent"])
        print("Predicted:", row["predicted_intent"])
        print("Helpfulness:", row["helpfulness_score"])
        print("Overall:", row["overall_score"])
        print("Decision:", row["decision"])
        print("Response:", row["generated_response"])
        print("-" * 70)

    # ---------------------------------------------------------
    # 3. Low overall quality
    # ---------------------------------------------------------
    print("\n\nLowest overall quality cases")
    print("=" * 70)

    low_overall = df.sort_values(
        "overall_score",
        ascending=True
    ).head(10)

    for _, row in low_overall.iterrows():
        print("\nGolden ID:", row["golden_id"])
        print("Customer:", row["customer_text"])
        print("Expected:", row["expected_intent"])
        print("Predicted:", row["predicted_intent"])
        print("Overall:", row["overall_score"])
        print("Helpfulness:", row["helpfulness_score"])
        print("Response:", row["generated_response"])
        print("-" * 70)


if __name__ == "__main__":
    main()