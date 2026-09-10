import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/processed/apple_conversations.csv"

MODEL_NAME = "all-MiniLM-L6-v2"


# --------------------------------------------------
# Intent detection
# --------------------------------------------------

def detect_intent(text):
    text = str(text).lower()

    if any(x in text for x in [
        "battery", "charging", "charger", "charge"
    ]):
        return "battery_charging"

    if any(x in text for x in [
        "apple id", "icloud", "password",
        "login", "sign in", "activation lock"
    ]):
        return "account_access"

    if any(x in text for x in [
        "wifi", "wi-fi", "bluetooth", "lte",
        "cellular", "internet", "network"
    ]):
        return "connectivity"

    if any(x in text for x in [
        "ios", "update", "updated", "upgrade",
        "software"
    ]):
        return "software_update"

    if any(x in text for x in [
        "apple music", "itunes", "podcast",
        "podcasts", "apple tv"
    ]):
        return "services_media"

    if any(x in text for x in [
        "payment", "billing", "refund",
        "charged", "credit card", "card"
    ]):
        return "billing_payment"

    if any(x in text for x in [
        "order", "shipping", "delivery",
        "reservation"
    ]):
        return "order_purchase"

    if any(x in text for x in [
        "app", "apps", "download",
        "install", "crash"
    ]):
        return "app_issue"

    if any(x in text for x in [
        "screen", "camera", "speaker",
        "button", "keyboard", "display"
    ]):
        return "hardware_device"

    return "general_troubleshooting"


# --------------------------------------------------
# Load data
# --------------------------------------------------

print("=" * 70)
print("INTENT-AWARE SIMILAR-CASE RETRIEVAL")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=["customer_text", "support_text"]
).copy()

print(f"Historical interactions: {len(df):,}")


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# Create embeddings
# --------------------------------------------------

print("Creating embeddings...")

embeddings = model.encode(
    df["customer_text"].tolist(),
    show_progress_bar=True,
    batch_size=64,
    normalize_embeddings=True,
)

print("Embeddings ready.")


# --------------------------------------------------
# Retrieve similar cases
# --------------------------------------------------

def retrieve_similar_cases(query, top_k=3):

    predicted_intent = detect_intent(query)

    print(
        f"\nDetected intent: {predicted_intent}"
    )

    # Create query embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    # Calculate semantic similarity
    scores = cosine_similarity(
        query_embedding,
        embeddings,
    )[0]

    # --------------------------------------------------
    # Retrieve top candidate pool
    # --------------------------------------------------

    candidate_count = min(
        20,
        len(df)
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


    # --------------------------------------------------
    # Intent-aware filtering
    # --------------------------------------------------

    candidates["candidate_intent"] = (
        candidates["customer_text"]
        .apply(detect_intent)
    )

    matching = candidates[
        candidates["candidate_intent"]
        == predicted_intent
    ].copy()


    # --------------------------------------------------
    # Response-quality filtering
    # --------------------------------------------------

    bad_reply_terms = [
        "here's how to customize",
        "see a photo you love",
    ]

    matching["support_lower"] = (
        matching["support_text"]
        .fillna("")
        .str.lower()
    )

    for term in bad_reply_terms:

        matching = matching[
            ~matching["support_lower"].str.contains(
                term,
                regex=False,
                na=False,
            )
        ]


    # --------------------------------------------------
    # Intent-specific response relevance
    # --------------------------------------------------

    intent_terms = {

        "battery_charging": [
            "battery",
            "charge",
            "charging",
        ],

        "connectivity": [
            "wifi",
            "wi-fi",
            "bluetooth",
            "lte",
            "cellular",
            "network",
        ],

        "account_access": [
            "icloud",
            "apple id",
            "password",
            "login",
            "sign in",
        ],

        "software_update": [
            "ios",
            "update",
            "software",
        ],

        "app_issue": [
            "app",
            "download",
            "install",
            "crash",
        ],

        "hardware_device": [
            "screen",
            "camera",
            "speaker",
            "button",
            "display",
        ],

        "billing_payment": [
            "payment",
            "billing",
            "refund",
            "charged",
            "card",
        ],

        "order_purchase": [
            "order",
            "shipping",
            "delivery",
        ],

        "services_media": [
            "music",
            "itunes",
            "podcast",
            "apple tv",
        ],
    }

    terms = intent_terms.get(
        predicted_intent,
        []
    )


    def response_relevance(text):

        text = str(text).lower()

        return sum(
            term in text
            for term in terms
        )


    matching["response_relevance"] = (
        matching["support_text"]
        .apply(response_relevance)
    )


    # --------------------------------------------------
    # Final ranking
    # --------------------------------------------------

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


    # --------------------------------------------------
    # Select final results
    # --------------------------------------------------

    if len(matching) >= top_k:

        results = matching.head(top_k)

    else:

        results = candidates.head(top_k)


    return results


# --------------------------------------------------
# Test
# --------------------------------------------------

query = input(
    "\nEnter a customer message:\n> "
)

results = retrieve_similar_cases(
    query,
    top_k=3,
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n" + "=" * 70)
print("TOP RELEVANT SUPPORT CASES")
print("=" * 70)


for i, (_, row) in enumerate(
    results.iterrows(),
    start=1,
):

    print(f"\nCASE {i}")
    print("-" * 70)

    print(
        f"Similarity       : "
        f"{row['similarity']:.4f}"
    )

    print(
        f"Detected intent  : "
        f"{row['candidate_intent']}"
    )

    print(
        f"Response relevance: "
        f"{row.get('response_relevance', 'N/A')}"
    )

    print("\nCUSTOMER:")
    print(row["customer_text"])

    print("\nAPPLE SUPPORT:")
    print(row["support_text"])