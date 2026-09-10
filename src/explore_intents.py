import pandas as pd
from collections import Counter
import re

INPUT_PATH = "data/processed/apple_conversations.csv"

print("Loading conversation dataset...\n")

df = pd.read_csv(INPUT_PATH)

print(f"Total interactions: {len(df):,}")

texts = df["customer_text"].fillna("").str.lower()

# Remove URLs
texts = texts.str.replace(r"http\S+", "", regex=True)

# Extract words
words = []

for text in texts:
    words.extend(
        re.findall(r"\b[a-z]{3,}\b", text)
    )

# Common words
counter = Counter(words)

# Common support-related words
stop_words = {
    "the", "and", "for", "this", "that", "with",
    "you", "are", "was", "but", "have", "has",
    "not", "from", "can", "just", "its", "they",
    "apple", "support", "please", "help", "why",
    "what", "how", "when", "does", "did",
    "www", "http", "com"
}

filtered = Counter({
    word: count
    for word, count in counter.items()
    if word not in stop_words
})

print("\n" + "=" * 70)
print("MOST COMMON CUSTOMER WORDS")
print("=" * 70)

for word, count in filtered.most_common(50):
    print(f"{word:25} {count:,}")