import pandas as pd
from collections import Counter

CSV_PATH = "data/raw/twcs.zip/twcs/twcs.csv"

brand_outbound = Counter()
total_rows = 0

print("Starting dataset scan...\n")

for chunk in pd.read_csv(CSV_PATH, chunksize=100_000):

    total_rows += len(chunk)

    # outbound = brand/support account messages
    outbound = chunk[chunk["inbound"] == False]

    brand_outbound.update(outbound["author_id"])

    if total_rows % 500_000 == 0:
        print(f"Processed {total_rows:,} rows...")

print("\n" + "=" * 60)
print("TOP SUPPORT ACCOUNTS")
print("=" * 60)

for brand, count in brand_outbound.most_common(30):
    print(f"{brand:30} {count:,}")

print("\n" + "=" * 60)
print(f"TOTAL ROWS PROCESSED: {total_rows:,}")
print("=" * 60)