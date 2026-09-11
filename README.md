# Hiver SDE Intern - AI Customer Support Agent

An AI-powered customer support agent built using the Customer Support on Twitter dataset.

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
