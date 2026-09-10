import pandas as pd

INPUT_PATH = "data/processed/apple_conversations.csv"

df = pd.read_csv(INPUT_PATH)

# Clean customer text
texts = (
    df["customer_text"]
    .fillna("")
    .str.lower()
    .str.replace(r"http\S+", "", regex=True)
    .str.replace(r"@\w+", "", regex=True)
    .str.strip()
)

groups = {
    "software_update": [
        r"\bios\b",
        r"\bupdate\b",
        r"\bupdated\b",
        r"\bupgrade\b",
        r"\bsoftware\b",
    ],
    "battery_charging": [
        r"\bbattery\b",
        r"\bcharge\b",
        r"\bcharging\b",
        r"\bcharger\b",
    ],
    "apps": [
        r"\bapp\b",
        r"\bapps\b",
        r"\bdownload\b",
        r"\binstall\b",
        r"\buninstall\b",
        r"\bcrash\b",
    ],
    "hardware": [
        r"\bscreen\b",
        r"\bbutton\b",
        r"\bcamera\b",
        r"\bspeaker\b",
        r"\bkeyboard\b",
        r"\bhome button\b",
    ],
    "account": [
        r"\bapple id\b",
        r"\bicloud\b",
        r"\bpassword\b",
        r"\blogin\b",
        r"\blog in\b",
        r"\baccount\b",
    ],
    "connectivity": [
        r"\bwifi\b",
        r"\bwi-fi\b",
        r"\binternet\b",
        r"\bbluetooth\b",
        r"\bnetwork\b",
        r"\bcellular\b",
    ],
    "payments": [
        r"\bpayment\b",
        r"\bbilling\b",
        r"\brefund\b",
        r"\bpurchase\b",
        r"\bcredit\b",
        r"\bcharged\b",
    ],
    "orders": [
        r"\border\b",
        r"\bdelivery\b",
        r"\bshipping\b",
        r"\breserved\b",
    ],
    "services_media": [
        r"\bapple music\b",
        r"\bitunes\b",
        r"\bicloud\b",
        r"\bapple tv\b",
        r"\bmusic\b",
        r"\bvideo\b",
    ],
}

print("=" * 70)
print("CORRECTED EXPLORATORY INTENT GROUPS")
print("=" * 70)

for group, keywords in groups.items():

    pattern = "|".join(keywords)

    mask = texts.str.contains(
        pattern,
        case=False,
        na=False,
        regex=True,
    )

    count = int(mask.sum())

    print(f"\n{group:20} {count:>7,}")

    print("  Examples:")

    examples = df.loc[mask, "customer_text"].head(3)

    for example in examples:
        print(f"   - {example[:180]}")