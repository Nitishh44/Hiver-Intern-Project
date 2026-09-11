import pandas as pd
from sklearn.metrics import cohen_kappa_score


def main():
    llm_path = "data/golden/llm_judge_results.csv"
    human_path = "data/golden/human_response_evaluation.csv"

    llm = pd.read_csv(llm_path)
    human = pd.read_csv(human_path)

    # Rename judge columns before merging
    llm = llm.rename(columns={
        "relevance": "relevance_llm",
        "groundedness": "groundedness_llm",
        "helpfulness": "helpfulness_llm",
        "overall": "overall_llm",
    })

    human = human.rename(columns={
    "relevance_score": "relevance_human",
    "groundedness_score": "groundedness_human",
    "helpfulness_score": "helpfulness_human",
    "overall_score": "overall_human",
    })

    df = pd.merge(
        llm,
        human,
        on="golden_id",
        how="inner",
        suffixes=("", "_dup")
    )

    print(f"Matched examples: {len(df)}")

    metrics = ["relevance", "groundedness", "helpfulness", "overall"]

    print("\nJudge Comparison")
    print("=" * 60)

    for col in metrics:
        llm_col = f"{col}_llm"
        human_col = f"{col}_human"

        # Remove invalid/missing values
        valid = df[[llm_col, human_col]].dropna()

        llm_scores = valid[llm_col].astype(int)
        human_scores = valid[human_col].astype(int)

        exact_agreement = (llm_scores == human_scores).mean() * 100
        mae = (llm_scores - human_scores).abs().mean()

        kappa = cohen_kappa_score(
            human_scores,
            llm_scores,
            weights="quadratic"
        )

        print(f"\n{col.upper()}")
        print(f"  Exact agreement : {exact_agreement:.2f}%")
        print(f"  Mean abs diff   : {mae:.2f}")
        print(f"  Quadratic kappa : {kappa:.3f}")

    print("\nAverage Scores")
    print("=" * 60)

    for col in metrics:
        llm_avg = df[f"{col}_llm"].mean()
        human_avg = df[f"{col}_human"].mean()

        print(
            f"{col.capitalize():15s} "
            f"LLM={llm_avg:.2f} | "
            f"Human={human_avg:.2f}"
        )


if __name__ == "__main__":
    main()