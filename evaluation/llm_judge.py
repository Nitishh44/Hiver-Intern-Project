import os
import json
import time
import pandas as pd
from google import genai
from google.genai import types


MODEL = "gemini-3.5-flash-lite"

INPUT_FILE = "data/golden/response_evaluation.csv"
OUTPUT_FILE = "data/golden/llm_judge_results.csv"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


INTENTS = [
    "software_update",
    "battery_charging",
    "app_issue",
    "hardware_device",
    "account_access",
    "connectivity",
    "billing_payment",
    "order_purchase",
    "services_media",
    "general_troubleshooting",
]


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "groundedness": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "helpfulness": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "overall": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "reason": {
            "type": "string",
        },
    },
    "required": [
        "relevance",
        "groundedness",
        "helpfulness",
        "overall",
        "reason",
    ],
}


def build_prompt(row):
    return f"""
You are evaluating an AI customer-support agent for an AppleSupport
customer-support dataset.

Evaluate ONLY the generated response using the customer message,
predicted intent, and retrieved historical evidence.

Important:
- Do not reward the system simply because the response sounds polite.
- Do not assume facts that are not supported by the retrieved evidence.
- Groundedness means the response does not invent unsupported claims.
- Relevance means it addresses the customer's actual issue.
- Helpfulness means it gives a useful next step or resolution direction.
- Overall is your overall quality assessment.
- Use the full 1-5 scale.

INTENT TAXONOMY:
{", ".join(INTENTS)}

CUSTOMER MESSAGE:
{row["customer_text"]}

EXPECTED INTENT:
{row["expected_intent"]}

PREDICTED INTENT:
{row["predicted_intent"]}

INTENT CONFIDENCE:
{row["intent_confidence"]}

INTENT MARGIN:
{row["intent_margin"]}

DECISION:
{row["decision"]}

DECISION REASON:
{row["decision_reason"]}

GENERATED RESPONSE:
{row["generated_response"]}

RETRIEVED REFERENCE CUSTOMER MESSAGE:
{row["top_reference_customer"]}

RETRIEVED REFERENCE SUPPORT RESPONSE:
{row["top_reference_support"]}

RETRIEVAL SIMILARITY:
{row["top_similarity"]}

Return ONLY the requested JSON object.
"""


def judge_one(row):
    response = client.models.generate_content(
        model=MODEL,
        contents=build_prompt(row),
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=JUDGE_SCHEMA,
            max_output_tokens=300,
        ),
    )

    return json.loads(response.text)


def load_existing_results():
    if not os.path.exists(OUTPUT_FILE):
        return pd.DataFrame()

    try:
        existing = pd.read_csv(OUTPUT_FILE)

        if "golden_id" not in existing.columns:
            return pd.DataFrame()

        return existing.drop_duplicates("golden_id")

    except Exception:
        return pd.DataFrame()


def main():
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it in PowerShell first."
        )

    df = pd.read_csv(INPUT_FILE)

    existing = load_existing_results()

    completed_ids = set()

    if not existing.empty:
        completed_ids = set(existing["golden_id"].astype(str))

    pending = df[
        ~df["golden_id"].astype(str).isin(completed_ids)
    ].copy()

    print(f"Total examples       : {len(df)}")
    print(f"Already completed    : {len(completed_ids)}")
    print(f"Remaining             : {len(pending)}")
    print(f"Model                 : {MODEL}")
    print()

    new_results = []

    for i, (_, row) in enumerate(pending.iterrows(), start=1):

        # Free tier limit is 15 requests/minute.
        # Keep requests safely below that limit.
        if i > 1:
            print("Waiting 5 seconds before next request...")
            time.sleep(5)

        try:
            judgment = judge_one(row)

            result = {
                "golden_id": row["golden_id"],
                "relevance": judgment["relevance"],
                "groundedness": judgment["groundedness"],
                "helpfulness": judgment["helpfulness"],
                "overall": judgment["overall"],
                "reason": judgment["reason"],
                "judge_model": MODEL,
            }

            new_results.append(result)

            print(
                f"Example {row['golden_id']} "
                f"| relevance={judgment['relevance']} "
                f"| grounded={judgment['groundedness']} "
                f"| helpful={judgment['helpfulness']} "
                f"| overall={judgment['overall']}"
            )

        except Exception as e:
            print(
                f"Example {row['golden_id']} FAILED: {e}"
            )

    # Combine old + new results.
    if new_results:
        new_df = pd.DataFrame(new_results)
    else:
        new_df = pd.DataFrame()

    if not existing.empty and not new_df.empty:
        final_df = pd.concat(
            [existing, new_df],
            ignore_index=True
        )
    elif not existing.empty:
        final_df = existing
    else:
        final_df = new_df

    if not final_df.empty:
        final_df = final_df.drop_duplicates(
            "golden_id",
            keep="last"
        ).sort_values("golden_id")

    final_df.to_csv(OUTPUT_FILE, index=False)

    print()
    print(
        f"Valid judgments: {len(final_df)}/{len(df)}"
    )
    print(f"Saved: {OUTPUT_FILE}")

    if len(final_df) > 0:
        print()
        print("Average scores:")
        print(
            final_df[
                [
                    "relevance",
                    "groundedness",
                    "helpfulness",
                    "overall",
                ]
            ].mean().round(2)
        )


if __name__ == "__main__":
    main()