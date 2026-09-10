import pandas as pd

CSV_PATH = "data/raw/twcs.zip/twcs/twcs.csv"
BRAND = "AppleSupport"

print("Loading AppleSupport sample...\n")

df = pd.read_csv(CSV_PATH, nrows=100_000)

# Keep AppleSupport tweets + customer tweets mentioning AppleSupport
apple = df[
    (df["author_id"] == BRAND)
    | (
        (df["inbound"] == True)
        & df["text"].str.contains(
            f"@{BRAND}",
            case=False,
            na=False,
            regex=False,
        )
    )
].copy()

# tweet_id -> complete row
tweet_map = {
    int(row["tweet_id"]): row
    for _, row in apple.iterrows()
}

print(f"AppleSupport-related rows: {len(apple):,}")

print("\n" + "=" * 70)
print("MULTI-TURN CONVERSATION EXAMPLES")
print("=" * 70)

shown = 0

for _, row in apple.iterrows():

    # Start with a customer message
    if row["inbound"] != True:
        continue

    current_id = int(row["tweet_id"])

    conversation = []

    # Walk backwards through parent tweets
    while current_id in tweet_map:

        current = tweet_map[current_id]

        conversation.append(current)

        parent_id = current["in_response_to_tweet_id"]

        if pd.isna(parent_id):
            break

        current_id = int(parent_id)

        if len(conversation) >= 8:
            break

    # Need at least 2 messages to call it a conversation
    if len(conversation) < 2:
        continue

    # Reverse because we walked backwards
    conversation.reverse()

    print("\n" + "-" * 70)

    for message in conversation:

        if message["inbound"]:
            speaker = "CUSTOMER"
        else:
            speaker = "APPLE SUPPORT"

        print(f"\n{speaker}:")
        print(message["text"])

    shown += 1

    if shown >= 5:
        break

print("\n" + "=" * 70)
print(f"CONVERSATIONS SHOWN: {shown}")
print("=" * 70)