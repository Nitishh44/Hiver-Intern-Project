# Hiver SDE Intern — AI Customer Support Agent

An AI-powered customer support agent built using the Customer Support on Twitter dataset.

The system:
- Classifies customer messages into support intents
- Retrieves similar historical support interactions
- Generates grounded support responses
- Decides whether to auto-handle or escalate to a human
- Evaluates classification, retrieval, response quality, and escalation behavior

## Target Brand

**AppleSupport**

AppleSupport was selected after comparing support-message volume, response coverage, customer diversity, and conversation structure.

## Running the System

After installing dependencies and placing the dataset at the required location:

```bash
python src/build_conversations.py

## Project Structure

```text
Hiver-SDE-Intern/
├── data/
│   ├── raw/
│   ├── processed/
│   └── golden/
├── src/
├── evaluation/
├── notebooks/
├── tests/
└── reports/