# Decision Log

## 1. Selected AppleSupport as the target brand
I compared candidate brands using customer-message volume, support-response coverage, unique customer count, and conversation structure. AppleSupport provided a strong balance of scale, response coverage, customer diversity, and intent diversity.

## 2. Reconstructed customer-support interactions using tweet relationships
Instead of treating every tweet independently, I used `in_response_to_tweet_id` and response relationships to reconstruct customer-to-support interactions. This provides better context for retrieval and evaluation.

## 3. Used one primary intent per message
Each customer message receives exactly one primary intent. The label represents the main problem the customer is trying to solve rather than every topic mentioned in the message. This makes evaluation deterministic.

## 4. Created a 10-intent taxonomy
I selected ten practical support intents: software update, battery/charging, app issues, hardware/device, account access, connectivity, billing/payment, order/purchase, services/media, and general troubleshooting. The taxonomy balances coverage with classification simplicity.

## 5. Used a golden evaluation set instead of relying on weak labels
The historical dataset does not provide the exact intent taxonomy required by the system. Therefore, I created a manually labeled 198-example golden set for the main evaluation.

## 6. Sampled the golden set for intent and difficulty coverage
The golden set was not designed as a prevalence estimate. Sampling intentionally included examples from different intent categories and difficult cases so that the evaluation tests system behavior across the taxonomy.

## 7. Used majority classification as the trivial baseline
The majority-class baseline establishes a minimum reference point. Macro-F1 is reported alongside accuracy because accuracy alone can hide poor performance on minority intents.

## 8. Used TF-IDF as the simple learned baseline
TF-IDF provides a lightweight lexical baseline that is substantially simpler than semantic embeddings. Comparing against it shows whether semantic retrieval/classification provides meaningful improvement.

## 9. Used sentence embeddings for semantic intent classification
I used `all-MiniLM-L6-v2` to compare customer messages with intent descriptions using cosine similarity. This allows semantically similar messages to match even when exact keywords differ.

## 10. Added intent margin to escalation decisions
Top-1 confidence alone can hide ambiguity. I therefore used the difference between the top two intent scores as an additional safety signal. A margin below 0.05 triggers escalation.

## 11. Added retrieval leakage protection
During evaluation, I excluded the evaluated tweet, its exact customer-text duplicate, and related identifiers from retrieval candidates. This prevents the system from retrieving the same example it is being evaluated on.

## 12. Used deterministic grounded responses instead of the local FLAN-T5 generator
A local FLAN-T5 experiment produced generic and unreliable responses. I therefore chose a constrained response layer focused on factuality, consistency, and safe escalation rather than presenting poor generative output as successful.

## 13. Treated LLM-as-judge as a secondary signal
The Gemini judge was validated against a 30-example manual/reference review. Agreement was weak, particularly for groundedness, so LLM-judge scores are not treated as ground truth.

## 14. Chose a conservative auto-handle policy
Auto-handling requires sufficient confidence, intent separation, retrieval evidence, and a safe intent. Ambiguous or weakly supported cases are escalated to a human to reduce the risk of incorrect automated responses.

## 15. Reported failure modes instead of optimizing only headline metrics
The evaluation revealed related-intent confusion, generic responses, context-dependent failures, misleading retrieval similarity, and missing conversation-state detection. These are reported explicitly because aggregate accuracy alone does not capture system reliability.