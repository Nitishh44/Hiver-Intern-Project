import os
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/processed/apple_conversations.csv"
EMBEDDINGS_PATH = "data/processed/apple_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


print("=" * 70)
print("LOADING RETRIEVAL SYSTEM")
print("=" * 70)

print("\nLoading conversation data...")
df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["customer_text", "support_text"]).reset_index(drop=True)

print(f"Conversation pairs loaded: {len(df):,}")

print("\nLoading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("Embedding model loaded!")


if os.path.exists(EMBEDDINGS_PATH):
    print("\nLoading cached embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Embeddings loaded: {embeddings.shape}")

else:
    print("\nCreating embeddings...")
    embeddings = model.encode(
        df["customer_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    np.save(EMBEDDINGS_PATH, embeddings)

    print(f"Embeddings saved: {embeddings.shape}")


INTENT_TERMS = {
    "software_update": [
        "update",
        "ios",
        "upgrade",
        "downgrade",
        "software",
        "version",
    ],
    "battery_charging": [
        "battery",
        "charge",
        "charging",
        "charger",
        "power",
        "drain",
    ],
    "app_issue": [
        "app",
        "application",
        "crash",
        "download",
        "install",
        "opening",
    ],
    "hardware_device": [
        "screen",
        "button",
        "camera",
        "speaker",
        "keyboard",
        "microphone",
        "device",
        "iphone",
        "ipad",
        "macbook",
    ],
    "account_access": [
        "apple id",
        "icloud",
        "password",
        "login",
        "account",
        "sign in",
        "authentication",
    ],
    "connectivity": [
        "wifi",
        "wi-fi",
        "bluetooth",
        "cellular",
        "network",
        "internet",
        "mobile data",
        "connection",
    ],
    "billing_payment": [
        "charge",
        "charged",
        "billing",
        "refund",
        "payment",
        "subscription",
        "purchase",
        "money",
    ],
    "order_purchase": [
        "order",
        "shipping",
        "delivery",
        "delivered",
        "reservation",
        "purchase",
    ],
    "services_media": [
        "music",
        "itunes",
        "apple tv",
        "podcast",
        "podcasts",
        "media",
        "playlist",
        "song",
        "video",
    ],
    "general_troubleshooting": [],
}


BAD_REPLY_TERMS = [
    "here's how to customize",
    "see a photo you love",
]


def normalize_text(text):
    """
    Normalize text for duplicate detection.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = re.sub(r"\s+", " ", text)

    return text


def clean_text(text):
    """
    Remove Twitter handles and URLs before keyword matching.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def response_relevance(customer_text, support_text, intent):
    """
    Lightweight heuristic to determine whether the historical
    support response is relevant to the customer's problem.
    """

    customer_clean = clean_text(customer_text)
    support_clean = clean_text(support_text)

    if not support_clean:
        return 0

    # Reject known irrelevant/template responses.
    for bad_term in BAD_REPLY_TERMS:
        if bad_term in support_clean:
            return 0

    terms = INTENT_TERMS.get(intent, [])

    if not terms:
        return 1

    customer_terms = set()

    for term in terms:
        if term in customer_clean:
            customer_terms.add(term)

    if not customer_terms:
        return 1

    matched = sum(
        1
        for term in customer_terms
        if term in support_clean
    )

    return matched


def retrieve_similar_cases(
    query,
    intent,
    top_k=3,
    exclude_tweet_ids=None,
    exclude_customer_texts=None,
):
    """
    Retrieve similar historical customer-support cases.

    During evaluation we exclude:
    1. The exact evaluation tweet ID.
    2. Exact duplicate customer messages.

    This prevents evaluation leakage.
    """

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings,
    )[0]

    scores = similarities.copy()

    # ---------------------------------------------------------
    # 1. Exclude specific tweet IDs
    # ---------------------------------------------------------

    if exclude_tweet_ids:

        excluded_ids = {
            str(tweet_id)
            for tweet_id in exclude_tweet_ids
        }

        if "customer_tweet_id" in df.columns:

            for index, tweet_id in enumerate(
                df["customer_tweet_id"]
            ):
                if str(tweet_id) in excluded_ids:
                    scores[index] = -1

        if "support_tweet_id" in df.columns:

            for index, tweet_id in enumerate(
                df["support_tweet_id"]
            ):
                if str(tweet_id) in excluded_ids:
                    scores[index] = -1

    # ---------------------------------------------------------
    # 2. Exclude exact duplicate customer texts
    # ---------------------------------------------------------

    if exclude_customer_texts:

        excluded_texts = {
            normalize_text(text)
            for text in exclude_customer_texts
            if normalize_text(text)
        }

        for index, customer_text in enumerate(
            df["customer_text"]
        ):

            normalized_customer_text = normalize_text(
                customer_text
            )

            if normalized_customer_text in excluded_texts:
                scores[index] = -1

    # ---------------------------------------------------------
    # 3. Get candidate pool
    # ---------------------------------------------------------

    candidate_indices = np.argsort(scores)[::-1]

    candidate_indices = [
        index
        for index in candidate_indices
        if scores[index] >= 0
    ]

    # We inspect more candidates because some may fail
    # intent/relevance filtering.
    candidate_indices = candidate_indices[:50]

    results = []

    # ---------------------------------------------------------
    # 4. Intent-aware filtering
    # ---------------------------------------------------------

    terms = INTENT_TERMS.get(intent, [])

    query_clean = clean_text(query)

    for index in candidate_indices:

        row = df.iloc[index]

        customer_text = str(row["customer_text"])
        support_text = str(row["support_text"])

        customer_clean = clean_text(customer_text)

        # -----------------------------------------------------
        # Intent filtering
        # -----------------------------------------------------

        if terms:

            has_intent_signal = any(
                term in customer_clean
                for term in terms
            )

            if not has_intent_signal:

                # Semantic similarity can still rescue a case
                # if it is extremely close.
                if scores[index] < 0.70:
                    continue

        # -----------------------------------------------------
        # Response quality filtering
        # -----------------------------------------------------

        relevance = response_relevance(
            customer_text,
            support_text,
            intent,
        )

        if relevance == 0:
            continue

        results.append(
            {
                "customer_text": customer_text,
                "support_text": support_text,
                "customer_tweet_id": row.get(
                    "customer_tweet_id",
                    None,
                ),
                "support_tweet_id": row.get(
                    "support_tweet_id",
                    None,
                ),
                "similarity": float(scores[index]),
                "response_relevance": relevance,
            }
        )

    # ---------------------------------------------------------
    # 5. Ranking
    # ---------------------------------------------------------

    results.sort(
        key=lambda item: (
            item["response_relevance"],
            item["similarity"],
        ),
        reverse=True,
    )

    return results[:top_k]


if __name__ == "__main__":

    test_query = (
        "My iPhone battery is draining very quickly "
        "after the latest iOS update"
    )

    test_intent = "battery_charging"

    print("\n" + "=" * 70)
    print("RETRIEVAL TEST")
    print("=" * 70)

    cases = retrieve_similar_cases(
        test_query,
        test_intent,
        top_k=3,
    )

    for i, case in enumerate(cases, start=1):

        print(f"\nCASE {i}")
        print("-" * 70)

        print(
            f"Similarity: "
            f"{case['similarity']:.4f}"
        )

        print(
            f"Response relevance: "
            f"{case['response_relevance']}"
        )

        print(
            f"\nCustomer:\n"
            f"{case['customer_text']}"
        )

        print(
            f"\nSupport:\n"
            f"{case['support_text']}"
        )