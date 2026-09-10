import os

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/apple_conversations.csv"
EMBEDDINGS_PATH = "data/processed/apple_embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("INTENT-AWARE SIMILAR-CASE RETRIEVAL")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH
)

df = df.dropna(
    subset=[
        "customer_text",
        "support_text",
    ]
).copy()

print(
    f"Historical interactions: {len(df):,}"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print(
    "Embedding model loaded!"
)


# ============================================================
# LOAD OR CREATE EMBEDDINGS
# ============================================================

if os.path.exists(
    EMBEDDINGS_PATH
):

    print(
        "\nLoading cached embeddings..."
    )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    print(
        "Cached embeddings loaded!"
    )

else:

    print(
        "\nNo cached embeddings found."
    )

    print(
        "Creating embeddings..."
    )

    embeddings = model.encode(
        df["customer_text"].tolist(),
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )

    print(
        "\nEmbeddings saved to:"
    )

    print(
        EMBEDDINGS_PATH
    )


print(
    f"Embedding shape: {embeddings.shape}"
)


# ============================================================
# INTENT-SPECIFIC RESPONSE RELEVANCE
# ============================================================

INTENT_TERMS = {

    "battery_charging": [
        "battery",
        "charge",
        "charging",
        "charger",
    ],

    "connectivity": [
        "wifi",
        "wi-fi",
        "bluetooth",
        "lte",
        "cellular",
        "network",
        "internet",
    ],

    "account_access": [
        "icloud",
        "apple id",
        "password",
        "login",
        "sign in",
        "account",
    ],

    "software_update": [
        "ios",
        "update",
        "updated",
        "upgrade",
        "software",
    ],

    "app_issue": [
        "app",
        "apps",
        "download",
        "install",
        "crash",
    ],

    "hardware_device": [
        "screen",
        "camera",
        "speaker",
        "button",
        "keyboard",
        "display",
        "microphone",
    ],

    "billing_payment": [
        "payment",
        "billing",
        "refund",
        "charged",
        "credit card",
        "card",
    ],

    "order_purchase": [
        "order",
        "shipping",
        "delivery",
        "reservation",
        "purchase",
    ],

    "services_media": [
        "music",
        "itunes",
        "podcast",
        "podcasts",
        "apple tv",
        "media",
    ],

    "general_troubleshooting": [],
}


# ============================================================
# BAD RESPONSE FILTER
# ============================================================

BAD_REPLY_TERMS = [
    "here's how to customize",
    "see a photo you love",
]


# ============================================================
# RETRIEVE SIMILAR CASES
# ============================================================

def retrieve_similar_cases(
    query,
    intent,
    top_k=3,
):

    print(
        f"\nRetrieval intent: {intent}"
    )


    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )


    # --------------------------------------------------------
    # Calculate semantic similarity
    # --------------------------------------------------------

    scores = cosine_similarity(
        query_embedding,
        embeddings,
    )[0]


    # --------------------------------------------------------
    # Retrieve larger candidate pool first
    # --------------------------------------------------------

    candidate_count = min(
        50,
        len(df),
    )

    top_indices = scores.argsort()[
        -candidate_count:
    ][::-1]


    candidates = df.iloc[
        top_indices
    ].copy()


    candidates["similarity"] = scores[
        top_indices
    ]


    # --------------------------------------------------------
    # Candidate intent filtering
    #
    # IMPORTANT:
    # We use the same semantic intent taxonomy indirectly
    # through lightweight text signals only for historical
    # candidate filtering.
    # --------------------------------------------------------

    terms = INTENT_TERMS.get(
        intent,
        [],
    )


    def candidate_matches_intent(text):

        text = str(text).lower()

        if not terms:
            return True

        return any(
            term in text
            for term in terms
        )


    candidates[
        "candidate_matches_intent"
    ] = candidates[
        "customer_text"
    ].apply(
        candidate_matches_intent
    )


    matching = candidates[
        candidates[
            "candidate_matches_intent"
        ]
    ].copy()


    # --------------------------------------------------------
    # Response quality filtering
    # --------------------------------------------------------

    matching["support_lower"] = (
        matching["support_text"]
        .fillna("")
        .str.lower()
    )


    for term in BAD_REPLY_TERMS:

        matching = matching[
            ~matching[
                "support_lower"
            ].str.contains(
                term,
                regex=False,
                na=False,
            )
        ]


    # --------------------------------------------------------
    # Response relevance
    # --------------------------------------------------------

    def response_relevance(
        text
    ):

        text = str(text).lower()

        return sum(
            term in text
            for term in terms
        )


    matching[
        "response_relevance"
    ] = matching[
        "support_text"
    ].apply(
        response_relevance
    )


    # --------------------------------------------------------
    # Final ranking
    # --------------------------------------------------------

    matching = matching.sort_values(
        by=[
            "response_relevance",
            "similarity",
        ],
        ascending=[
            False,
            False,
        ],
    )


    # --------------------------------------------------------
    # Return only genuinely matching cases
    # --------------------------------------------------------

    if len(matching) == 0:

        print(
            "No intent-matching historical cases found."
        )

        return matching


    results = matching.head(
        top_k
    ).copy()


    return results


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    query = input(
        "\nEnter a customer message:\n> "
    )


    # Temporary manual intent for testing retrieval.
    # The real agent will pass the semantic classifier's
    # predicted intent automatically.

    intent = input(
        "\nEnter detected intent:\n> "
    )


    results = retrieve_similar_cases(
        query,
        intent,
        top_k=3,
    )


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP RELEVANT SUPPORT CASES"
    )

    print(
        "=" * 70
    )


    if results.empty:

        print(
            "\nNo relevant support cases found."
        )


    else:

        for i, (_, row) in enumerate(
            results.iterrows(),
            start=1,
        ):

            print(
                f"\nCASE {i}"
            )

            print(
                "-" * 70
            )

            print(
                f"Similarity        : "
                f"{row['similarity']:.4f}"
            )

            print(
                f"Response relevance: "
                f"{row['response_relevance']}"
            )

            print(
                "\nCUSTOMER:"
            )

            print(
                row["customer_text"]
            )

            print(
                "\nAPPLE SUPPORT:"
            )

            print(
                row["support_text"]
            )