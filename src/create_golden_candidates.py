import pandas as pd
import re

INPUT_PATH = "data/processed/apple_conversations.csv"
OUTPUT_PATH = "data/golden/golden_candidates.csv"

df = pd.read_csv(INPUT_PATH)

# ---------------------------------------------------------
# Clean obvious acknowledgement-only messages
# ---------------------------------------------------------

text = (
    df["customer_text"]
    .fillna("")
    .str.lower()
    .str.strip()
)

# Remove @mentions and URLs for acknowledgement detection
clean_text = (
    text
    .str.replace(r"@\w+", "", regex=True)
    .str.replace(r"http\S+", "", regex=True)
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

ack_pattern = (
    r"^(thanks?|thank you|thx|"
    r"awesome thanks?|great thanks?|"
    r"worked|it worked|"
    r"okay thanks?|ok thanks?|"
    r"yes|no|yep|yeah)$"
)

df = df[
    ~clean_text.str.match(
        ack_pattern,
        na=False
    )
].copy()

# ---------------------------------------------------------
# Remove duplicate customer messages
# ---------------------------------------------------------

df = df.drop_duplicates(
    subset=["customer_text"]
).copy()

# ---------------------------------------------------------
# Reproducible candidate sample
# ---------------------------------------------------------

sample_size = min(1000, len(df))

candidates = df.sample(
    n=sample_size,
    random_state=42
).copy()

# Empty annotation fields
candidates["intent"] = ""
candidates["annotation_notes"] = ""

candidates.to_csv(
    OUTPUT_PATH,
    index=False
)

print("=" * 70)
print("CLEAN GOLDEN SET CANDIDATE POOL")
print("=" * 70)

print(f"Remaining interactions : {len(df):,}")
print(f"Candidate pool         : {len(candidates):,}")
print(f"Saved to               : {OUTPUT_PATH}")

print("\nSample:")
print(
    candidates[
        ["customer_text", "support_text", "intent"]
    ].head(10).to_string(index=False)
)