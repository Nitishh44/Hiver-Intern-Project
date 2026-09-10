import pandas as pd
from collections import Counter

CSV_PATH = "data/raw/twcs.zip/twcs/twcs.csv"
BRAND = "AppleSupport"

customer_messages = []
customer_authors = set()

total_brand_replies = 0
total_customer_messages = 0

print("Analyzing AppleSupport conversations...\n")

for chunk in pd.read_csv(CSV_PATH, chunksize=100_000):

    # Apple support replies
    replies = chunk[
        (chunk["inbound"] == False)
        & (chunk["author_id"] == BRAND)
    ]

    total_brand_replies += len(replies)

    # Customer tweets directed at AppleSupport
    mask = (
        (chunk["inbound"] == True)
        & chunk["text"].str.contains(
            "@AppleSupport",
            case=False,
            na=False,
            regex=False,
        )
    )

    customers = chunk[mask]

    total_customer_messages += len(customers)

    customer_authors.update(
        customers["author_id"].dropna().tolist()
    )

    # Keep a sample for manual inspection
    if len(customer_messages) < 100:
        customer_messages.extend(
            customers["text"].tolist()
        )


print("=" * 70)
print("APPLE SUPPORT DATA QUALITY")
print("=" * 70)

print(f"Customer messages : {total_customer_messages:,}")
print(f"Support replies   : {total_brand_replies:,}")
print(f"Unique customers  : {len(customer_authors):,}")

print("\n" + "=" * 70)
print("SAMPLE CUSTOMER MESSAGES")
print("=" * 70)

for i, text in enumerate(customer_messages[:50], 1):
    print(f"\n{i}. {text}")