import os
import sys

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src",
)

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from response_generator import run_support_agent


# ============================================================
# PATHS
# ============================================================

GOLDEN_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "golden",
    "golden_set.csv",
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "golden",
    "response_evaluation.csv",
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 30
RANDOM_SEED = 42


# ============================================================
# LOAD GOLDEN SET
# ============================================================

print("=" * 70)
print("RESPONSE QUALITY EVALUATION")
print("=" * 70)

golden = pd.read_csv(
    GOLDEN_PATH
)

print(
    f"\nTotal golden examples: "
    f"{len(golden)}"
)


# ============================================================
# SELECT EVALUATION SAMPLE
# ============================================================

sample_size = min(
    SAMPLE_SIZE,
    len(golden),
)

evaluation_sample = golden.sample(
    n=sample_size,
    random_state=RANDOM_SEED,
).reset_index(drop=True)

print(
    f"Response evaluation examples: "
    f"{len(evaluation_sample)}"
)


# ============================================================
# GOLDEN TWEET IDS
# ============================================================

golden_tweet_ids = set(
    str(tweet_id)
    for tweet_id in golden[
        "customer_tweet_id"
    ].dropna()
)

print(
    f"Golden tweet IDs available for "
    f"exclusion: {len(golden_tweet_ids)}"
)


# ============================================================
# EVALUATION
# ============================================================

results = []

leakage_count = 0

no_retrieval_count = 0


for i, row in evaluation_sample.iterrows():

    golden_id = row[
        "golden_id"
    ]

    customer_message = str(
        row["customer_text"]
    )

    expected_intent = str(
        row["intent"]
    )

    golden_tweet_id = str(
        row["customer_tweet_id"]
    )

    print(
        f"\nProcessing {i + 1}/"
        f"{len(evaluation_sample)}"
    )

    print(
        f"Customer: "
        f"{customer_message[:150]}"
    )


    # --------------------------------------------------------
    # Run support agent with leakage protection
    # --------------------------------------------------------

    result = run_support_agent(
        customer_message,

        exclude_tweet_ids={
            golden_tweet_id
        },

        exclude_customer_texts={
            customer_message
        },
    )


    # --------------------------------------------------------
    # Extract result
    # --------------------------------------------------------

    predicted_intent = result[
        "intent"
    ]

    confidence = float(
        result["confidence"]
    )

    second_intent = result[
        "second_intent"
    ]

    second_score = float(
        result["second_score"]
    )

    intent_margin = float(
        result["intent_margin"]
    )

    generated_response = result[
        "response"
    ]

    decision = result[
        "decision"
    ]

    reason = result[
        "reason"
    ]

    retrieved_cases = result[
        "retrieved_cases"
    ]


    # --------------------------------------------------------
    # Retrieval information
    # --------------------------------------------------------

    if not retrieved_cases:

        no_retrieval_count += 1

        top_similarity = None

        top_reference_customer = ""

        top_reference_support = ""

        retrieved_tweet_ids = []

    else:

        top_case = retrieved_cases[0]

        top_similarity = float(
            top_case["similarity"]
        )

        top_reference_customer = str(
            top_case["customer_text"]
        )

        top_reference_support = str(
            top_case["support_text"]
        )

        retrieved_tweet_ids = [
            str(
                case.get(
                    "customer_tweet_id",
                    "",
                )
            )
            for case in retrieved_cases
        ]


    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    own_tweet_retrieved = (
        golden_tweet_id
        in retrieved_tweet_ids
    )


    # Also check exact duplicate customer text.
    duplicate_customer_retrieved = False

    normalized_customer = (
        customer_message.strip().lower()
    )


    for case in retrieved_cases:

        retrieved_customer = str(
            case.get(
                "customer_text",
                "",
            )
        ).strip().lower()

        if (
            retrieved_customer
            == normalized_customer
        ):

            duplicate_customer_retrieved = True

            break


    leakage_detected = (
        own_tweet_retrieved
        or duplicate_customer_retrieved
    )


    if leakage_detected:

        leakage_count += 1

        print(
            "WARNING: Retrieval leakage detected!"
        )

    else:

        print(
            "Leakage check: PASS"
        )


    # --------------------------------------------------------
    # Print model result
    # --------------------------------------------------------

    print(
        f"Expected intent : "
        f"{expected_intent}"
    )

    print(
        f"Predicted intent: "
        f"{predicted_intent}"
    )

    print(
        f"Confidence      : "
        f"{confidence:.4f}"
    )

    print(
        f"Second intent   : "
        f"{second_intent}"
    )

    print(
        f"Second score    : "
        f"{second_score:.4f}"
    )

    print(
        f"Intent margin   : "
        f"{intent_margin:.4f}"
    )

    print(
        f"Decision        : "
        f"{decision}"
    )


    if top_similarity is not None:

        print(
            f"Top similarity  : "
            f"{top_similarity:.4f}"
        )

    else:

        print(
            "Top similarity  : None"
        )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append(
        {
            "golden_id": golden_id,

            "customer_text": customer_message,

            "expected_intent": expected_intent,

            "predicted_intent": predicted_intent,

            "intent_correct": (
                predicted_intent
                == expected_intent
            ),

            "intent_confidence": confidence,

            "second_intent": second_intent,

            "second_score": second_score,

            "intent_margin": intent_margin,

            "generated_response": (
                generated_response
            ),

            "decision": decision,

            "decision_reason": reason,

            "top_similarity": (
                top_similarity
            ),

            "top_reference_customer": (
                top_reference_customer
            ),

            "top_reference_support": (
                top_reference_support
            ),

            "retrieved_customer_tweet_ids": (
                ", ".join(
                    retrieved_tweet_ids
                )
            ),

            "leakage_detected": (
                leakage_detected
            ),

            "duplicate_customer_retrieved": (
                duplicate_customer_retrieved
            ),

            "own_tweet_retrieved": (
                own_tweet_retrieved
            ),
        }
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
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RESPONSE EVALUATION SUMMARY")
print("=" * 70)

print(
    f"\nExamples evaluated : "
    f"{len(results_df)}"
)

print(
    f"Leakage detected   : "
    f"{leakage_count}"
)

print(
    f"No retrieval cases : "
    f"{no_retrieval_count}"
)

print(
    f"Leakage-free cases : "
    f"{len(results_df) - leakage_count}"
)


if len(results_df) > 0:

    intent_accuracy = (
        results_df[
            "intent_correct"
        ].mean()
        * 100
    )

    print(
        f"\nIntent accuracy on "
        f"response sample: "
        f"{intent_accuracy:.2f}%"
    )


# ============================================================
# RETRIEVAL STATISTICS
# ============================================================

retrieval_scores = results_df[
    "top_similarity"
].dropna()


if len(retrieval_scores) > 0:

    print(
        f"Average top retrieval "
        f"similarity: "
        f"{retrieval_scores.mean():.4f}"
    )

    print(
        f"Minimum top retrieval "
        f"similarity: "
        f"{retrieval_scores.min():.4f}"
    )

    print(
        f"Maximum top retrieval "
        f"similarity: "
        f"{retrieval_scores.max():.4f}"
    )


# ============================================================
# INTENT MARGIN STATISTICS
# ============================================================

if "intent_margin" in results_df.columns:

    print(
        f"\nAverage intent margin: "
        f"{results_df['intent_margin'].mean():.4f}"
    )

    print(
        f"Minimum intent margin: "
        f"{results_df['intent_margin'].min():.4f}"
    )

    print(
        f"Maximum intent margin: "
        f"{results_df['intent_margin'].max():.4f}"
    )


# ============================================================
# DECISION DISTRIBUTION
# ============================================================

print(
    "\nDecision distribution:"
)

print(
    results_df[
        "decision"
    ].value_counts()
)


# ============================================================
# OUTPUT
# ============================================================

print(
    f"\nResults saved to:"
)

print(
    OUTPUT_PATH
)

print("\nDone!")