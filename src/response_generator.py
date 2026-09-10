import re

from retrieval import retrieve_similar_cases
from semantic_intent_classifier import classify_intent


# ============================================================
# GROUNDED RESPONSE GENERATOR
# ============================================================

def clean_support_text(text):
    """
    Remove Twitter handles and URLs from historical replies.
    """

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+",
        "",
        text,
    )

    # Remove Twitter handles
    text = re.sub(
        r"@\w+",
        "",
        text,
    )

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# RESPONSE TEMPLATES
# ============================================================

def generate_grounded_response(
    customer_message,
    intent,
    retrieved_cases,
):

    if retrieved_cases.empty:

        return (
            "We'd be happy to help with this issue. "
            "Please send us a DM with more details so "
            "we can look into it further."
        )


    # Use the strongest historical support response.
    best_case = retrieved_cases.iloc[0]

    historical_response = clean_support_text(
        best_case["support_text"]
    )


    # --------------------------------------------------------
    # Intent-specific grounded responses
    # --------------------------------------------------------

    if intent == "battery_charging":

        return (
            "We understand how frustrating battery issues "
            "can be. We'd be happy to troubleshoot this with "
            "you. Please send us a DM with more details about "
            "your iPhone so we can look into it further."
        )


    if intent == "connectivity":

        return (
            "We understand you're having connectivity issues. "
            "We'd be happy to look into this with you. "
            "Please send us a DM with more details so we can "
            "troubleshoot the issue further."
        )


    if intent == "app_issue":

        return (
            "We're sorry you're having trouble with the app. "
            "We'd be happy to help troubleshoot the issue. "
            "Please send us a DM with more details so we can "
            "look into it further."
        )


    if intent == "hardware_device":

        return (
            "We're sorry you're experiencing an issue with "
            "your device. We'd be happy to look into this "
            "with you. Please send us a DM with more details "
            "so we can troubleshoot the issue."
        )


    if intent == "software_update":

        return (
            "We understand you're having trouble after the "
            "software update. We'd be happy to look into "
            "what's happening. Please send us a DM with "
            "more details so we can troubleshoot further."
        )


    if intent == "services_media":

        return (
            "We're sorry you're having trouble with the "
            "service. We'd be happy to look into this with "
            "you. Please send us a DM with more details so "
            "we can troubleshoot the issue."
        )


    if intent == "account_access":

        return (
            "We understand you're having trouble accessing "
            "your account. Please send us a DM with more "
            "details so our support team can securely look "
            "into the issue with you."
        )


    if intent == "billing_payment":

        return (
            "We understand your concern about the billing "
            "or payment issue. Please send us a DM with "
            "more details so our support team can securely "
            "look into this with you."
        )


    if intent == "order_purchase":

        return (
            "We understand your concern about your order. "
            "Please send us a DM with the relevant details "
            "so our support team can look into the issue "
            "with you."
        )


    # General troubleshooting
    return (
        "We're sorry you're experiencing this issue. "
        "We'd be happy to look into it with you. "
        "Please send us a DM with more details so we "
        "can troubleshoot the issue further."
    )


# ============================================================
# ESCALATION DECISION
# ============================================================

def escalation_decision(
    intent,
    confidence,
    retrieved_cases,
):

    # Sensitive cases should be reviewed by a human.
    high_risk_intents = {
        "billing_payment",
        "account_access",
        "order_purchase",
    }


    if intent in high_risk_intents:

        return (
            "ESCALATE_TO_HUMAN",
            "Sensitive issue involving account, billing, "
            "or order information.",
        )


    # No evidence available.
    if retrieved_cases.empty:

        return (
            "ESCALATE_TO_HUMAN",
            "No relevant historical support cases found.",
        )


    top_similarity = float(
        retrieved_cases.iloc[0]["similarity"]
    )


    # Low semantic intent confidence.
    if confidence < 0.30:

        return (
            "ESCALATE_TO_HUMAN",
            "Intent classification confidence is low.",
        )


    # Strong historical evidence.
    if top_similarity >= 0.75:

        return (
            "AUTO_HANDLE",
            "High similarity with relevant historical "
            "support cases.",
        )


    return (
        "ESCALATE_TO_HUMAN",
        "Retrieval confidence is below the "
        "auto-handle threshold.",
    )


# ============================================================
# COMPLETE SUPPORT AGENT
# ============================================================

def run_support_agent(
    customer_message,
):

    # --------------------------------------------------------
    # 1. Semantic intent classification
    # --------------------------------------------------------

    intent, confidence, ranked_intents = (
        classify_intent(
            customer_message
        )
    )


    # --------------------------------------------------------
    # 2. Intent-aware retrieval
    # --------------------------------------------------------

    retrieved_cases = retrieve_similar_cases(
        customer_message,
        intent,
        top_k=3,
    )


    # --------------------------------------------------------
    # 3. Grounded response generation
    # --------------------------------------------------------

    response = generate_grounded_response(
        customer_message,
        intent,
        retrieved_cases,
    )


    # --------------------------------------------------------
    # 4. Escalation decision
    # --------------------------------------------------------

    decision, reason = escalation_decision(
        intent,
        confidence,
        retrieved_cases,
    )


    return {
        "intent": intent,
        "confidence": confidence,
        "response": response,
        "decision": decision,
        "reason": reason,
        "retrieved_cases": retrieved_cases,
        "ranked_intents": ranked_intents,
    }


# ============================================================
# INTERACTIVE TEST
# ============================================================

if __name__ == "__main__":

    customer_message = input(
        "\nEnter a customer message:\n> "
    )


    result = run_support_agent(
        customer_message
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print("AI SUPPORT AGENT RESULT")
    print("=" * 70)


    print(
        f"\nIntent     : {result['intent']}"
    )

    print(
        f"Confidence : {result['confidence']:.4f}"
    )

    print(
        f"Decision   : {result['decision']}"
    )

    print(
        f"Reason     : {result['reason']}"
    )


    # ========================================================
    # AI RESPONSE
    # ========================================================

    print("\nAI RESPONSE:")
    print("-" * 70)

    print(
        result["response"]
    )


    # ========================================================
    # REFERENCE CASES
    # ========================================================

    print("\n" + "=" * 70)
    print("REFERENCE CASES")
    print("=" * 70)


    for i, (_, row) in enumerate(
        result["retrieved_cases"].iterrows(),
        start=1,
    ):

        print(
            f"\nCASE {i}"
        )

        print(
            "-" * 70
        )

        print(
            f"Similarity: "
            f"{row['similarity']:.4f}"
        )

        print(
            f"Customer: "
            f"{row['customer_text']}"
        )

        print(
            f"Support: "
            f"{row['support_text']}"
        )