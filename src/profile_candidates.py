import pandas as pd
from collections import Counter

CSV_PATH = "data/raw/twcs.zip/twcs/twcs.csv"

CANDIDATES = {
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
}

stats = {
    brand: {
        "outbound": 0,
        "inbound": 0,
        "unique_customers": set(),
        "customer_messages_with_response": 0,
        "customer_messages_with_parent": 0,
    }
    for brand in CANDIDATES
}

total_rows = 0

print("Starting candidate profiling...\n")

for chunk in pd.read_csv(CSV_PATH, chunksize=100_000):
    total_rows += len(chunk)

    for brand in CANDIDATES:
        # Brand/support messages
        brand_outbound = chunk[
            (chunk["inbound"] == False)
            & (chunk["author_id"] == brand)
        ]

        stats[brand]["outbound"] += len(brand_outbound)

        # Customer messages mentioning this brand
        mask = (
            (chunk["inbound"] == True)
            & chunk["text"].str.contains(
                f"@{brand}",
                case=False,
                na=False,
                regex=False,
            )
        )

        customer = chunk[mask]

        stats[brand]["inbound"] += len(customer)

        stats[brand]["unique_customers"].update(
            customer["author_id"].dropna().tolist()
        )

        stats[brand]["customer_messages_with_response"] += (
            customer["response_tweet_id"].notna().sum()
        )

        stats[brand]["customer_messages_with_parent"] += (
            customer["in_response_to_tweet_id"].notna().sum()
        )

    if total_rows % 500_000 == 0:
        print(f"Processed {total_rows:,} rows...")


print("\n" + "=" * 75)
print("CANDIDATE BRAND COMPARISON")
print("=" * 75)

print(
    f"{'Brand':20}"
    f"{'Inbound':>12}"
    f"{'Outbound':>12}"
    f"{'Customers':>12}"
    f"{'Has Reply':>12}"
    f"{'Has Parent':>12}"
)

print("-" * 75)

for brand, s in stats.items():

    reply_rate = (
        s["customer_messages_with_response"] / s["inbound"]
        if s["inbound"]
        else 0
    )

    parent_rate = (
        s["customer_messages_with_parent"] / s["inbound"]
        if s["inbound"]
        else 0
    )

    print(
        f"{brand:20}"
        f"{s['inbound']:>12,}"
        f"{s['outbound']:>12,}"
        f"{len(s['unique_customers']):>12,}"
        f"{reply_rate:>11.1%}"
        f"{parent_rate:>11.1%}"
    )

print("\nTotal rows scanned:", f"{total_rows:,}")