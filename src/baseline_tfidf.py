import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


DATA_PATH = "data/processed/apple_conversations.csv"
GOLDEN_PATH = "data/golden/golden_set.csv"
PREDICTION_PATH = "data/golden/tfidf_predictions.csv"


print("=" * 70)
print("BASELINE #2 — TF-IDF + LOGISTIC REGRESSION")
print("=" * 70)


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
golden = pd.read_csv(GOLDEN_PATH)

print(f"Conversation pairs : {len(df):,}")
print(f"Golden examples    : {len(golden):,}")


# --------------------------------------------------
# 2. Keep golden set completely out of training
# --------------------------------------------------

golden_ids = set(
    golden["customer_tweet_id"].dropna()
)

train_df = df[
    ~df["customer_tweet_id"].isin(golden_ids)
].copy()

print(
    f"Training pool      : {len(train_df):,}"
)


# --------------------------------------------------
# 3. Weak-label training data
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

print("\nWeak-label distribution:")
print(train_df["intent"].value_counts())


# --------------------------------------------------
# 4. TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=50_000,
)

X_train = vectorizer.fit_transform(
    train_df["customer_text"].fillna("")
)

y_train = train_df["intent"]


# --------------------------------------------------
# 5. Train classifier
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 6. Predict GOLDEN SET
# --------------------------------------------------

X_golden = vectorizer.transform(
    golden["customer_text"].fillna("")
)

y_true = golden["intent"]
y_pred = model.predict(X_golden)


# --------------------------------------------------
# 7. Evaluate
# --------------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
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

print(
    f"Accuracy : {accuracy:.4f} ({accuracy:.2%})"
)

print(
    f"Macro F1 : {macro_f1:.4f} ({macro_f1:.2%})"
)


print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )
)


# --------------------------------------------------
# 8. Save predictions for error analysis
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
    index=False
)


print("=" * 70)
print("PREDICTIONS SAVED")
print("=" * 70)

print(PREDICTION_PATH)