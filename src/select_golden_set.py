import pandas as pd

INPUT_PATH = "data/golden/golden_candidates.csv"
OUTPUT_PATH = "data/golden/golden_set.csv"

df = pd.read_csv(INPUT_PATH)

text = (
    df["customer_text"]
    .fillna("")
    .str.lower()
)

# Exploratory signals only.
signals = {
    "software_update": [
        "ios", "update", "updated", "upgrade", "software"
    ],
    "battery_charging": [
        "battery", "charge", "charging", "charger"
    ],
    "app_issue": [
        " app ", " apps ", "download", "install", "crash"
    ],
    "hardware_device": [
        "screen", "button", "camera", "speaker", "keyboard"
    ],
    "account_access": [
        "apple id", "icloud", "password", "login", "account"
    ],
    "connectivity": [
        "wifi", "wi-fi", "bluetooth", "internet",
        "network", "cellular"
    ],
    "billing_payment": [
        "payment", "billing", "refund", "purchase",
        "charged", "credit"
    ],
    "order_purchase": [
        "order", "delivery", "shipping", "reservation"
    ],
    "services_media": [
        "apple music", "itunes", "apple tv",
        "podcast", "music", "video"
    ],
}

# ---------------------------------------------------------
# Select up to 20 examples per exploratory category
# ---------------------------------------------------------

selected = []

for intent, keywords in signals.items():

    pattern = "|".join(
        keyword.replace(" ", r"\s+")
        for keyword in keywords
    )

    matches = df[
        text.str.contains(
            pattern,
            regex=True,
            na=False
        )
    ]

    # Reproducible selection
    matches = matches.sample(
        n=min(20, len(matches)),
        random_state=42
    )

    selected.append(matches)

# Combine category samples
golden = pd.concat(
    selected,
    ignore_index=True
)

# Remove duplicate tweets
golden = golden.drop_duplicates(
    subset=["customer_tweet_id"]
)

# Add extra random examples to reach 200
remaining = df[
    ~df["customer_tweet_id"].isin(
        golden["customer_tweet_id"]
    )
]

needed = 200 - len(golden)

if needed > 0:
    extra = remaining.sample(
        n=min(needed, len(remaining)),
        random_state=42
    )

    golden = pd.concat(
        [golden, extra],
        ignore_index=True
    )

# Shuffle final dataset
golden = golden.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Human annotation fields
golden["golden_id"] = range(1, len(golden) + 1)

golden["intent"] = ""
golden["annotation_notes"] = ""

# Keep exactly 200
golden = golden.head(200)

# Final column order
golden = golden[
    [
        "golden_id",
        "customer_text",
        "support_text",
        "customer_id",
        "customer_tweet_id",
        "support_tweet_id",
        "intent",
        "annotation_notes",
    ]
]

golden.to_csv(
    OUTPUT_PATH,
    index=False
)

print("=" * 70)
print("FINAL GOLDEN SET CREATED")
print("=" * 70)

print(f"Golden examples : {len(golden)}")
print(f"Saved to        : {OUTPUT_PATH}")

print("\nColumns:")
print(list(golden.columns))

print("\nFirst 10 examples:")
print(
    golden[
        ["customer_text", "support_text", "intent"]
    ].head(10).to_string(index=False)
)