import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

print("=" * 70)
print("SEMANTIC INTENT CLASSIFIER")
print("=" * 70)

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded!")


# ============================================================
# INTENT TAXONOMY
# ============================================================

INTENT_DESCRIPTIONS = {

    "software_update":
        "Problems with iOS or macOS software updates, "
        "upgrading, downgrading, update failures, or software versions.",

    "battery_charging":
        "Battery draining quickly, battery life problems, "
        "charging problems, charger or power issues.",

    "app_issue":
        "Problems with an application, including apps crashing, "
        "apps not working, downloading, installing, or opening apps.",

    "hardware_device":
        "Physical device problems such as screen, buttons, camera, "
        "speaker, keyboard, microphone, or other hardware malfunction.",

    "account_access":
        "Apple ID, iCloud, password, login, account access, "
        "account recovery, or authentication problems.",

    "connectivity":
        "Wi-Fi, Bluetooth, cellular network, mobile data, "
        "internet connection, or other connectivity problems.",

    "billing_payment":
        "Unexpected charges, billing problems, refunds, purchases, "
        "payment issues, subscriptions, or payment disputes.",

    "order_purchase":
        "Problems involving orders, purchases, shipping, delivery, "
        "order status, reservations, or receiving a product.",

    "services_media":
        "Problems with Apple Music, iTunes, Apple TV, podcasts, "
        "media playback, media libraries, or Apple media services.",

    "general_troubleshooting":
        "General Apple technical support that does not clearly "
        "fit another intent, or messages with insufficient information."
}


# ============================================================
# CREATE INTENT EMBEDDINGS
# ============================================================

intent_names = list(
    INTENT_DESCRIPTIONS.keys()
)

intent_descriptions = list(
    INTENT_DESCRIPTIONS.values()
)

print("\nCreating intent embeddings...")

intent_embeddings = model.encode(
    intent_descriptions,
    normalize_embeddings=True,
)

print("Intent embeddings created!")


# ============================================================
# CLASSIFY MESSAGE
# ============================================================

def classify_intent(customer_message):

    message_embedding = model.encode(
        [customer_message],
        normalize_embeddings=True,
    )

    similarities = cosine_similarity(
        message_embedding,
        intent_embeddings,
    )[0]

    # Rank all intents from highest to lowest similarity
    ranked_indices = np.argsort(
        similarities
    )[::-1]

    # --------------------------------------------------------
    # TOP-1 INTENT
    # --------------------------------------------------------

    best_index = int(
        ranked_indices[0]
    )

    predicted_intent = intent_names[
        best_index
    ]

    confidence = float(
        similarities[best_index]
    )

    # --------------------------------------------------------
    # TOP-2 INTENT
    # --------------------------------------------------------

    second_index = int(
        ranked_indices[1]
    )

    second_intent = intent_names[
        second_index
    ]

    second_score = float(
        similarities[second_index]
    )

    # --------------------------------------------------------
    # INTENT MARGIN
    # --------------------------------------------------------
    # Difference between top-1 and top-2.
    #
    # Larger margin = clearer classification
    # Smaller margin = more ambiguous classification
    # --------------------------------------------------------

    intent_margin = (
        confidence - second_score
    )

    # --------------------------------------------------------
    # RANKED INTENTS
    # --------------------------------------------------------

    ranked_intents = []

    for index in ranked_indices:

        ranked_intents.append(
            (
                intent_names[index],
                float(similarities[index]),
            )
        )

    return (
        predicted_intent,
        confidence,
        ranked_intents,
        second_intent,
        second_score,
        intent_margin,
    )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def detect_intent_semantic(customer_message):

    (
        intent,
        confidence,
        _,
        _,
        _,
        _,
    ) = classify_intent(
        customer_message
    )

    return intent


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    customer_message = input(
        "\nEnter a customer message:\n> "
    )

    (
        intent,
        confidence,
        ranked,
        second_intent,
        second_score,
        intent_margin,
    ) = classify_intent(
        customer_message
    )

    print("\n" + "=" * 70)
    print("SEMANTIC CLASSIFICATION RESULT")
    print("=" * 70)

    print(
        f"\nPredicted intent : {intent}"
    )

    print(
        f"Confidence       : {confidence:.4f}"
    )

    print(
        f"Second intent    : {second_intent}"
    )

    print(
        f"Second score     : {second_score:.4f}"
    )

    print(
        f"Intent margin    : {intent_margin:.4f}"
    )

    print("\nTop intent candidates:")
    print("-" * 70)

    for name, score in ranked[:5]:

        print(
            f"{name:25} {score:.4f}"
        )