# 1. Problem Framing & Approach

## Problem

The goal was to build an AI customer-support agent that can take an incoming customer message, identify the customer's primary support intent, retrieve similar historical support interactions, draft a grounded response, and decide whether the case can be handled automatically or should be escalated to a human.

I used the Customer Support on Twitter dataset and selected **AppleSupport** as the target support account.

## Why AppleSupport?

I compared candidate brands using:

- Customer-message volume
- Support-response coverage
- Unique customer count
- Conversation structure
- Intent diversity

AppleSupport provided a strong balance across these dimensions rather than simply being selected because it had the largest number of messages.

## End-to-End Approach

The system has four main stages:

1. **Intent classification**  
   Incoming messages are embedded using `all-MiniLM-L6-v2` and compared against descriptions of 10 manually defined support intents using cosine similarity.

2. **Historical retrieval**  
   The system retrieves semantically similar AppleSupport customer-support interactions from reconstructed historical conversations.

3. **Grounded response generation**  
   Retrieved cases are used as grounding context for a constrained support-response layer. A local FLAN-T5 experiment was also tested, but its responses were frequently generic or irrelevant, so it was not used in the final path.

4. **Escalation decision**  
   The agent considers intent risk, retrieval availability, classifier confidence, and the margin between the top two intent scores. Ambiguous cases are escalated to a human.

## Intent Taxonomy

I defined 10 primary intents:

- `software_update`
- `battery_charging`
- `app_issue`
- `hardware_device`
- `account_access`
- `connectivity`
- `billing_payment`
- `order_purchase`
- `services_media`
- `general_troubleshooting`

Each message receives **one primary intent**. The goal is to identify the customer's main problem rather than assigning every intent suggested by individual keywords.

## Conversation Reconstruction

The original dataset contains tweet-level relationships rather than ready-made support conversations. I reconstructed customer-support interactions using the available response relationships and retained the tweet IDs for traceability and leakage control.

This produced **36,658 linked AppleSupport customer-support pairs** for retrieval and downstream analysis.


# 2. Dataset & Golden Evaluation Set

## Dataset

The project uses the Customer Support on Twitter dataset. The raw dataset contains tweet-level customer and support interactions rather than clean conversation records.

For this project, I focused on the **AppleSupport** account.

After reconstructing the available response relationships, the pipeline produced:

- **36,658 linked customer-support pairs**
- Customer and support tweet IDs retained for traceability
- Generated embeddings used for semantic retrieval

The raw dataset and generated embedding cache are excluded from the Git repository.

## Golden Evaluation Set

To avoid evaluating the system only against noisy automatically generated labels, I created a dedicated golden evaluation set.

The final golden set contains **198 labeled examples** covering all 10 defined intents.

Examples were sampled with the goal of covering different intents and difficulty levels rather than preserving the original class distribution. Obvious acknowledgements and unusable examples were excluded during preparation.

## Golden Set Distribution

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

The distribution is intentionally not representative of production traffic. It was designed to ensure that less frequent intents were still evaluated.

## Evaluation Philosophy

The main classification benchmark is the untouched 198-example golden set.

For comparison, I used two baselines:

1. **Majority baseline** — always predicts the most frequent golden-set intent.
2. **TF-IDF baseline** — a simple learned text classifier trained using weak historical labels and evaluated on the untouched golden set.

The semantic embedding classifier is then compared against both baselines using accuracy and macro-F1.



# 3. Model & Baseline Results

## Baselines

I used two baselines to establish progressively stronger reference points.

### Majority Baseline

The majority baseline always predicts the most frequent intent in the golden evaluation set.

It achieves:

- Accuracy: **28.79%**
- Macro-F1: **4.47%**

This is intentionally trivial, but it establishes the minimum benchmark and shows why accuracy alone can be misleading on an imbalanced intent set.

### TF-IDF Baseline

The second baseline is a simple learned text classifier using TF-IDF features.

It achieves:

- Accuracy: **44.95%**
- Macro-F1: **51.12%**

The TF-IDF model provides a meaningful lexical baseline, but its performance is limited when messages use different wording for similar problems.

## Semantic Classifier

The final classifier uses the `all-MiniLM-L6-v2` sentence-transformer model.

For each customer message:

1. The message is converted into an embedding.
2. The embedding is compared with the embedding of each intent description.
3. The intent with the highest cosine similarity is selected.
4. The second-highest intent is retained to calculate an ambiguity margin.

The final semantic classifier achieves:

- Accuracy: **63.64%**
- Macro-F1: **59.36%**

### Comparison

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority baseline | 28.79% | 4.47% |
| TF-IDF baseline | 44.95% | 51.12% |
| **Semantic classifier** | **63.64%** | **59.36%** |

The semantic classifier improves accuracy by **18.69 percentage points** over TF-IDF and **34.85 percentage points** over the majority baseline.

The improvement over TF-IDF suggests that semantic similarity is useful for handling support messages whose wording differs from the historical examples used by the simpler lexical baseline.

## Per-Intent Observations

The classifier performs relatively well on `account_access`, `connectivity`, and `general_troubleshooting`.

The main weaknesses are intents with substantial semantic overlap, especially:

- `hardware_device` vs `software_update`
- `app_issue` vs `services_media`
- `billing_payment` vs `services_media`

This motivated the addition of the top-1/top-2 intent margin to the escalation policy.



# 4. Response Quality & Escalation Evaluation

## Response Evaluation

I evaluated 30 examples separately from the 198-example classification benchmark to inspect end-to-end support behavior.

The evaluation measured:

- Intent accuracy
- Retrieval similarity
- Response relevance
- Response groundedness
- Response helpfulness
- Overall response quality
- Retrieval leakage

### Results

| Metric | Result |
|---|---:|
| Intent accuracy | **63.33%** |
| Average retrieval similarity | **0.7586** |
| Relevance | **3.27 / 5** |
| Groundedness | **4.43 / 5** |
| Helpfulness | **2.80 / 5** |
| Overall quality | **2.83 / 5** |
| Retrieval leakage | **0 cases** |

The strongest result is groundedness at 4.43/5. However, helpfulness is only 2.80/5, showing that the system often produces safe but overly generic responses.

## Escalation Policy

The system does not rely on classification confidence alone.

A case is escalated when:

- The intent is considered high-risk
- No useful historical case is retrieved
- Classification confidence is below the minimum threshold
- The top two intent scores are too close
- Retrieval similarity is not strong enough for automatic handling

The **intent margin** is defined as:

```text
intent margin = top-1 intent score - top-2 intent score



# 5. Failure Analysis

I reviewed the response-evaluation examples to understand where the system fails rather than relying only on aggregate metrics.

## Top 5 Failure Modes

### 1. Closely Related Intents Are Confused

**Evidence:** 11 of 30 response-evaluation examples received the wrong intent.

Examples include:

- A screen-freezing problem classified as `software_update` instead of `hardware_device`
- An app-performance problem classified as `software_update`
- A payment-related Apple Music issue classified as `services_media` instead of `billing_payment`

**Hypothesis:** Semantic similarity captures related topics well, but it does not always distinguish the customer's primary problem.

**One-week fix:** Add contrastive intent examples and an intent-aware reranker.

### 2. Responses Are Too Generic

**Evidence:** 13 of 30 examples received a helpfulness score of 2 or below.

For example, a detailed customer message describing an iTunes download/login problem can receive a generic support response instead of a specific troubleshooting suggestion.

**Hypothesis:** The constrained response layer prioritizes safety and groundedness, but does not extract enough actionable information from retrieved cases.

**One-week fix:** Extract concrete troubleshooting steps and evidence from the best retrieved historical interactions.

### 3. Short and Context-Dependent Messages

Messages such as version updates, acknowledgements, and short follow-ups are difficult to classify correctly without conversation history.

**Hypothesis:** The current classifier primarily sees the current customer message and does not explicitly model recent conversation state.

**One-week fix:** Include the most recent conversation turns as additional classifier context.

### 4. High Retrieval Similarity Does Not Guarantee Correctness

Some incorrect predictions still have high retrieval similarity.

For example, one incorrectly classified hardware-related case had retrieval similarity around **0.87**.

**Hypothesis:** The retrieval system measures topical similarity rather than whether the retrieved case has the correct intent or represents the same underlying problem.

**One-week fix:** Rerank retrieved cases using predicted intent, intent margin, and conversation state.

### 5. No Explicit Conversation-State Detector

The current system does not explicitly distinguish between a new problem, a follow-up, an acknowledgement, or a resolved issue.

For example:

- `"I just installed the new update, I think it solved the issue. Thanks!"`
- `"I think I'm going to skip that one"`

can be treated as support requests even though they may represent a resolved or conversational state.

**Hypothesis:** Intent classification alone is insufficient for short conversational messages.

**One-week fix:** Add an explicit conversation-state classifier with states such as:

- `NEW_ISSUE`
- `FOLLOW_UP`
- `RESOLVED`
- `ACKNOWLEDGEMENT`
- `ESCALATION_REQUIRED`

## What These Failures Suggest

The main bottleneck is not retrieval coverage alone. The system needs better **conversation understanding**.

The highest-value improvements would therefore be:

1. Use conversation history during classification.
2. Add intent-aware retrieval reranking.
3. Detect conversation state explicitly.
4. Generate responses from concrete evidence in retrieved cases.
5. Replace the current prototype response layer with a more capable grounded generator once quality can be validated.



# 6. What Is Misleading About My Headline Number?

The headline result is **63.64% classification accuracy** on the 198-example golden set.

This number is useful, but it should not be interpreted as end-to-end support-agent accuracy or production readiness.

There are three important caveats:

1. **The golden set is small.**  
   It contains 198 examples, so the measured accuracy has uncertainty and may not generalize directly to production traffic.

2. **The sampling is not production-prevalence sampling.**  
   The golden set was intentionally constructed to cover intents and difficulty levels. Therefore, its class distribution does not represent the true distribution of AppleSupport customer requests.

3. **Classification accuracy does not measure response quality.**  
   On the separate 30-example response evaluation, overall response quality was only **2.83 / 5** and helpfulness was **2.80 / 5**.

Therefore, the more accurate headline is:

> **The semantic classifier achieved 63.64% accuracy and 59.36% Macro-F1 on a 198-example golden evaluation set, while the end-to-end response layer still has significant quality limitations.**

This distinction is important because a good classification score alone does not imply that an automated support agent is ready to respond to customers without human oversight.

# 7. One-More-Week Plan

If given one additional week, I would prioritize improvements based on the observed failure modes.

## Priority 1 — Add Conversation Context

Pass the recent conversation turns to the classifier instead of classifying the current tweet in isolation.

**Expected benefit:** better handling of short follow-ups, acknowledgements and context-dependent messages.

## Priority 2 — Improve Intent Classification

Add contrastive training/examples for frequently confused intents such as:

- `hardware_device` vs `software_update`
- `app_issue` vs `services_media`
- `billing_payment` vs `services_media`

Then evaluate an intent-aware reranker.

**Expected benefit:** reduce closely related intent errors.

## Priority 3 — Improve Retrieval

Rerank retrieved cases using:

- Intent compatibility
- Semantic similarity
- Conversation state
- Historical response quality

**Expected benefit:** make retrieval similarity more predictive of useful support evidence rather than merely topical similarity.

## Priority 4 — Improve Response Generation

Replace the current deterministic response layer with a grounded generator that can extract concrete troubleshooting steps from retrieved cases.

The generator should be constrained to retrieved evidence and should abstain when sufficient evidence is unavailable.

**Expected benefit:** improve the current 2.80/5 helpfulness score without sacrificing groundedness.

## Priority 5 — Add Conversation-State Detection

Introduce explicit states:

```text
NEW_ISSUE
FOLLOW_UP
RESOLVED
ACKNOWLEDGEMENT
ESCALATION_REQUIRED