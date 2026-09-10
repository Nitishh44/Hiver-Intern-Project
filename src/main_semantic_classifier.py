import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


DATA_PATH = "data/processed/apple_conversations.csv"
GOLDEN_PATH = "data/golden/golden_set.csv"
PREDICTION_PATH = "data/golden/semantic_predictions.csv"

MODEL_NAME = "all-MiniLM-L6-v2"


print("=" * 70)
print("MAIN MODEL — SEMANTIC EMBEDDINGS + LOGISTIC REGRESSION")
print("=" * 70)


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
golden = pd.read_csv(GOLDEN_PATH)

golden_ids = set(golden["customer_tweet_id"].dropna())

train_df = df[
    ~df["customer_tweet_id"].isin(golden_ids)
].copy()

print(f"Conversation pairs : {len(df):,}")
print(f"Training examples  : {len(train_df):,}")
print(f"Golden examples    : {len(golden):,}")


# --------------------------------------------------
# 2. Weak-label training data
# --------------------------------------------------

def assign_intent(text):

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


train_df["intent"] = train_df[
    "customer_text"
].apply(assign_intent)

print("\nTraining labels:")
print(train_df["intent"].value_counts())


# --------------------------------------------------
# 3. Load embedding model
# --------------------------------------------------

print("\nLoading embedding model...")

encoder = SentenceTransformer(MODEL_NAME)

print("Embedding training data...")


# --------------------------------------------------
# 4. Create semantic embeddings
# --------------------------------------------------

X_train = encoder.encode(
    train_df["customer_text"].fillna("").tolist(),
    show_progress_bar=True,
    batch_size=64,
)

y_train = train_df["intent"]


# --------------------------------------------------
# 5. Train classifier
# --------------------------------------------------

print("\nTraining Logistic Regression...")

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)

classifier.fit(X_train, y_train)


# --------------------------------------------------
# 6. Encode golden set
# --------------------------------------------------

print("\nEmbedding golden set...")

X_golden = encoder.encode(
    golden["customer_text"].fillna("").tolist(),
    show_progress_bar=True,
    batch_size=64,
)


# --------------------------------------------------
# 7. Predict
# --------------------------------------------------

y_true = golden["intent"]
y_pred = classifier.predict(X_golden)


# --------------------------------------------------
# 8. Evaluate
# --------------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred,
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)


print("\n" + "=" * 70)
print("GOLDEN SET RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f} ({accuracy:.2%})")
print(f"Macro F1 : {macro_f1:.4f} ({macro_f1:.2%})")


print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )
)


# --------------------------------------------------
# 9. Save predictions
# --------------------------------------------------

predictions = golden[
    [
        "golden_id",
        "customer_text",
        "intent",
    ]
].copy()

predictions["predicted_intent"] = y_pred

predictions.to_csv(
    PREDICTION_PATH,
    index=False,
)


print("=" * 70)
print("PREDICTIONS SAVED")
print("=" * 70)

print(PREDICTION_PATH)