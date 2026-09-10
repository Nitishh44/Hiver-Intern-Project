import pandas as pd

CSV_PATH = "data/raw/twcs.zip/twcs/twcs.csv"
OUTPUT_PATH = "data/processed/apple_conversations.csv"
BRAND = "AppleSupport"

print("Building AppleSupport conversation pairs...\n")

# ---------------------------------------------------------
# Pass 1: Load only AppleSupport-related tweets
# ---------------------------------------------------------

apple_rows = []

for chunk in pd.read_csv(CSV_PATH, chunksize=100_000):

    support = chunk[
        (chunk["inbound"] == False)
        & (chunk["author_id"] == BRAND)
    ]

    customer = chunk[
        (chunk["inbound"] == True)
        & chunk["text"].str.contains(
            f"@{BRAND}",
            case=False,
            na=False,
            regex=False,
        )
    ]

    selected = pd.concat([support, customer])

    apple_rows.append(selected)

    print(f"Processed chunk — AppleSupport rows collected: "
          f"{sum(len(x) for x in apple_rows):,}")

apple = pd.concat(apple_rows, ignore_index=True)

print("\nTotal AppleSupport-related rows:", len(apple))

# ---------------------------------------------------------
# Build tweet ID lookup
# ---------------------------------------------------------

tweet_map = {
    int(row["tweet_id"]): row
    for _, row in apple.iterrows()
}

# ---------------------------------------------------------
# Build customer → support pairs
# ---------------------------------------------------------

pairs = []

for _, customer in apple.iterrows():

    if customer["inbound"] != True:
        continue

    parent_id = customer["in_response_to_tweet_id"]

    if pd.isna(parent_id):
        continue

    parent_id = int(parent_id)

    if parent_id not in tweet_map:
        continue

    support = tweet_map[parent_id]

    # Parent must be an AppleSupport reply
    if support["inbound"] == True:
        continue

    pairs.append({
        "customer_text": customer["text"],
        "support_text": support["text"],
        "customer_id": customer["author_id"],
        "customer_tweet_id": customer["tweet_id"],
        "support_tweet_id": support["tweet_id"],
    })

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

pairs_df = pd.DataFrame(pairs)

pairs_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("CONVERSATION DATASET CREATED")
print("=" * 70)

print(f"Linked interactions : {len(pairs_df):,}")
print(f"Saved to             : {OUTPUT_PATH}")

print("\nSample:")
print(pairs_df.head(10).to_string(index=False))