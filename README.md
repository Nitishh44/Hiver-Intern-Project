# Hiver SDE Intern - AI Customer Support Agent

An AI-powered customer support agent built on the Customer Support on Twitter dataset.

The system:

- Classifies incoming customer messages into one of 10 support intents
- Retrieves similar historical AppleSupport interactions
- Generates concise grounded support responses
- Decides whether to auto-handle or escalate to a human
- Evaluates classification, response quality, and escalation behavior

## Target Brand

**AppleSupport**

AppleSupport was selected after comparing candidate brands using customer-message volume, support-response coverage, unique customer count, and conversation structure. It provided a strong balance of scale, response coverage, customer diversity, and sufficient intent diversity for evaluation.

## System Overview

```text
Customer Message
       |
       v
Semantic Intent Classifier
       |
       +----> Top Intent
       |      Confidence
       |      Top-2 Margin
       |
       v
Historical Case Retrieval
       |
       v
Grounded Response Generator
       |
       v
Escalation Decision
       |
       +----> AUTO_HANDLE
       |
       +----> ESCALATE_TO_HUMAN
```

## Intent Taxonomy

The system assigns exactly one primary intent.

| Intent | Description |
|---|---|
| `software_update` | iOS/macOS updates, upgrades, downgrade or update failures |
| `battery_charging` | Battery drain, charging and power-related issues |
| `app_issue` | App crashes, apps not working, download/install issues |
| `hardware_device` | Device hardware problems such as screen, buttons, camera or keyboard |
| `account_access` | Apple ID, iCloud, password and login/access problems |
| `connectivity` | Wi-Fi, Bluetooth, cellular and network issues |
| `billing_payment` | Charges, refunds, billing and payment problems |
| `order_purchase` | Orders, shipping, delivery and purchases |
| `services_media` | Apple Music, iTunes, Apple TV, podcasts and media issues |
| `general_troubleshooting` | Technical support that does not fit the other intents |

The labeling rule is one primary intent per customer message. The primary problem is preferred over secondary keywords.

## Dataset Processing

The project uses the Kaggle Customer Support on Twitter dataset.

Raw data is intentionally excluded from GitHub.

The pipeline:

1. Profiles available support brands.
2. Selects AppleSupport.
3. Reconstructs customer-to-support interactions using tweet response relationships.
4. Produces **36,658 linked customer-support pairs**.
5. Creates a golden evaluation set of **198 hand-labeled examples**.

The golden set was sampled for intent and difficulty coverage rather than production prevalence.

## Evaluation Set

The final golden set contains 198 examples.

| Intent | Examples |
|---|---:|
| `general_troubleshooting` | 57 |
| `services_media` | 23 |
| `software_update` | 19 |
| `hardware_device` | 18 |
| `battery_charging` | 18 |
| `connectivity` | 18 |
| `account_access` | 18 |
| `app_issue` | 12 |
| `order_purchase` | 9 |
| `billing_payment` | 6 |

## Classification Results

Three approaches were evaluated on the untouched golden set.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority baseline | 28.79% | 4.47% |
| TF-IDF baseline | 44.95% | 51.12% |
| Semantic classifier | **63.64%** | **59.36%** |

The semantic classifier improves accuracy by **18.69 percentage points** over the TF-IDF baseline.

The semantic model uses `all-MiniLM-L6-v2` with intent descriptions and cosine similarity.

The reported confidence is a similarity score, not a calibrated probability.

## Retrieval

Historical AppleSupport interactions are embedded using `all-MiniLM-L6-v2`.

For each incoming message, the retriever:

- Finds semantically similar historical cases
- Applies intent-aware filtering
- Removes low-quality or unsuitable historical replies
- Excludes evaluation examples to prevent leakage
- Also excludes exact duplicate customer messages

Embeddings are cached locally under `data/processed/` and are ignored by Git.

## Response Generation

The response layer uses retrieved historical cases as grounding evidence and produces concise intent-specific support responses.

A local FLAN-T5 generation experiment was also tested. It frequently produced generic responses, so the final system uses a constrained deterministic response layer instead of presenting low-quality generated text as a successful result.

This prioritizes controllability and groundedness, but response helpfulness remains a limitation.

## Escalation Policy

The agent escalates when:

- The intent is high-risk
- No suitable retrieval result exists
- Intent confidence is below `0.30`
- The top-two intent margin is below `0.05`
- Otherwise, the case is auto-handled only when the top retrieval similarity is at least `0.75`

The intent margin is:

```text
top intent score - second intent score
```

A threshold experiment on the response-evaluation sample found:

| Margin threshold | Auto-handled | Escalated | Auto intent accuracy |
|---:|---:|---:|---:|
| 0.00 | 15 | 15 | 60.00% |
| 0.02 | 9 | 21 | 77.78% |
| 0.03 | 9 | 21 | 77.78% |
| **0.05** | **9** | **21** | **77.78%** |
| 0.08 | 7 | 23 | 71.43% |
| 0.10 | 5 | 25 | 60.00% |

The final threshold was set to `0.05` because it improved auto-handled intent accuracy without requiring unnecessarily aggressive escalation on this evaluation sample.

## Response Evaluation

A separate 30-example response evaluation measured retrieval and response quality.

| Metric | Result |
|---|---:|
| Intent accuracy | 63.33% |
| Retrieval similarity (average) | 0.7586 |
| Relevance | 3.27 / 5 |
| Groundedness | 4.43 / 5 |
| Helpfulness | 2.80 / 5 |
| Overall quality | 2.83 / 5 |
| Retrieval leakage | 0 |

The results show that grounding is relatively strong, while response helpfulness and specificity remain important weaknesses.

## LLM-as-Judge

An LLM judge was used as a secondary response-quality signal.

Agreement with the model-assisted manual/reference ratings was:

| Metric | Exact Agreement | Quadratic Kappa |
|---|---:|---:|
| Relevance | 23.33% | 0.327 |
| Groundedness | 20.00% | -0.004 |
| Helpfulness | 26.67% | 0.263 |
| Overall | 23.33% | 0.211 |

These ratings are not treated as ground truth. Agreement was weak, particularly for groundedness, so the LLM judge is reported as a supplementary signal rather than a replacement for manual evaluation.

## Top Failure Modes

1. **Closely related intents are confused**

   Hardware, app, software-update and services/billing issues can be semantically similar.

2. **Responses can be too generic**

   13 of 30 response-evaluation examples received helpfulness scores of 2 or below.

3. **Short messages depend on conversation context**

   Version updates, acknowledgements and follow-ups are difficult to interpret from a single tweet.

4. **High retrieval similarity does not guarantee correctness**

   Some incorrect classifications still retrieved highly similar historical examples.

5. **No explicit conversation-state detector**

   Resolved issues and acknowledgements can still be treated as support requests.

## What Is Misleading About My Headline Number?

The headline result is **63.64% classification accuracy** on the 198-example golden set.

This should not be interpreted as production or end-to-end support-agent accuracy.

The golden set was intentionally sampled for intent and difficulty coverage rather than production prevalence. It is also relatively small.

More importantly, classification accuracy does not measure the quality of the final support response. On the separate response evaluation, overall quality was **2.83 / 5** and helpfulness was **2.80 / 5**.

Therefore:

> The semantic classifier achieved 63.64% accuracy and 59.36% Macro-F1 on a 198-example golden evaluation set, while the end-to-end response layer still has significant quality limitations.

## Reproducibility

### 1. Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Add the dataset

Download the Customer Support on Twitter dataset and place the CSV at:

```text
data/raw/twcs.zip/twcs/twcs.csv
```

The raw dataset is not included in this repository.

### 3. Build AppleSupport interactions

```powershell
python src/build_conversations.py
```

This creates:

```text
data/processed/apple_conversations.csv
```

### 4. Build the golden-set candidates

```powershell
python src/create_golden_candidates.py
python src/select_golden_set.py
```

### 5. Run classification evaluation

```powershell
python evaluation/evaluate_intent.py
```

### 6. Run response evaluation

```powershell
python evaluation/evaluate_responses.py
python evaluation/evaluate_response_metrics.py
```

### 7. Run escalation evaluation

```powershell
python evaluation/evaluate_escalation.py
```

### 8. Run tests

```powershell
pytest -q
```

The repository contains the generated evaluation artifacts needed to inspect the reported results. Embeddings are generated locally and are not committed to Git.

## Project Structure

```text
Hiver-SDE-Intern/
|
├── data/
│   ├── raw/                  # Local dataset, not committed
│   ├── processed/            # Generated data/embeddings
│   └── golden/               # Evaluation datasets and results
|
├── src/
│   ├── profile_brands.py
│   ├── analyze_apple.py
│   ├── build_conversations.py
│   ├── semantic_intent_classifier.py
│   ├── baseline_majority.py
│   ├── baseline_tfidf.py
│   ├── retrieval.py
│   └── response_generator.py
|
├── evaluation/
│   ├── evaluate_intent.py
│   ├── evaluate_responses.py
│   ├── evaluate_response_metrics.py
│   ├── evaluate_escalation.py
│   ├── llm_judge.py
│   ├── compare_judges.py
│   └── failure_analysis.py
|
├── reports/
│   ├── decision_log.md
│   └── final_report.md
|
├── tests/
│   └── test_support_agent.py
|
├── README.md
└── requirements.txt
```

## Limitations

- The golden evaluation set contains 198 examples.
- Golden-set sampling is designed for coverage, not production prevalence.
- The response evaluation contains 30 examples.
- The current response layer is intentionally constrained and can be generic.
- The semantic confidence score is not calibrated.
- Retrieval is heuristic rather than learned end-to-end.
- Conversation context is not yet fully modeled.
- The LLM judge showed weak agreement with manual/reference ratings.
- The system is a research prototype and should not be considered production-ready.

## Detailed Report

For the complete methodology, decision log, evaluation analysis, failure examples and one-week improvement plan, see:

`reports/final_report.md`