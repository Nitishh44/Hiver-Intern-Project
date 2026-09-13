import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

SOURCE_PATH = "data/processed/apple_conversations.csv"

DEMO_DIR = "data/processed/demo"

DEMO_DATA_PATH = os.path.join(
    DEMO_DIR,
    "apple_demo_conversations.csv",
)

DEMO_EMBEDDINGS_PATH = os.path.join(
    DEMO_DIR,
    "apple_demo_embeddings.npy",
)

MODEL_NAME = "all-MiniLM-L6-v2"

DEMO_SIZE = 5000

RANDOM_SEED = 42


# ============================================================
# CREATE DIRECTORY
# ============================================================

os.makedirs(
    DEMO_DIR,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CREATING DEPLOYMENT DEMO INDEX")
print("=" * 70)

print("\nLoading conversation data...")

df = pd.read_csv(
    SOURCE_PATH
)

df = df.dropna(
    subset=[
        "customer_text",
        "support_text",
    ]
).reset_index(drop=True)

print(
    f"Available conversation pairs: {len(df):,}"
)


# ============================================================
# SAMPLE REPRESENTATIVE CASES
# ============================================================

if len(df) > DEMO_SIZE:

    demo_df = df.sample(
        n=DEMO_SIZE,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

else:

    demo_df = df.copy()


print(
    f"Demo conversation pairs: {len(demo_df):,}"
)


# ============================================================
# SAVE DEMO DATA
# ============================================================

demo_df.to_csv(
    DEMO_DATA_PATH,
    index=False,
)

print(
    f"\nSaved demo data to: {DEMO_DATA_PATH}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print("Embedding model loaded!")


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print(
    f"\nCreating embeddings for {len(demo_df):,} cases..."
)

embeddings = model.encode(
    demo_df["customer_text"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True,
)


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

np.save(
    DEMO_EMBEDDINGS_PATH,
    embeddings,
)

print(
    f"\nEmbeddings shape: {embeddings.shape}"
)

print(
    f"Saved embeddings to: {DEMO_EMBEDDINGS_PATH}"
)

print("\nDemo index creation complete.")