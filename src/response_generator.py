import re


# ============================================================
# IMPORTS
# ============================================================

try:
    from .retrieval import retrieve_similar_cases
    from .semantic_intent_classifier import classify_intent
except ImportError:
    from retrieval import retrieve_similar_cases
    from semantic_intent_classifier import classify_intent


# ============================================================
# CLEAN SUPPORT TEXT
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
# GROUNDED RESPONSE GENERATOR
# ============================================================

def generate_grounded_response(
    customer_message,
    intent,
    retrieved_cases,
):
    """
    Generate a concise support response based on the
    detected intent and retrieved historical cases.

    The final response is intentionally constrained rather
    than directly copying a historical response.
    """

    # --------------------------------------------------------
    # No historical evidence
    # --------------------------------------------------------

    if not retrieved_cases:

        return (
            "We'd be happy to help with this issue. "
            "Please send us a DM with more details so "
            "we can look into it further."
        )

    # Use the strongest historical support case as evidence.
    best_case = retrieved_cases[0]

    historical_response = clean_support_text(
        best_case.get(
            "support_text",
            "",
        )
    )

    # Keep the historical response available as grounding
    # evidence, but do not copy it directly.
    _ = historical_response


    # --------------------------------------------------------
    # Battery / charging
    # --------------------------------------------------------

    if intent == "battery_charging":

        return (
            "We understand how frustrating battery issues "
            "can be. We'd be happy to troubleshoot this with "
            "you. Please send us a DM with more details about "
            "your iPhone so we can look into it further."
        )


    # --------------------------------------------------------
    # Connectivity
    # --------------------------------------------------------

    if intent == "connectivity":

        return (
            "We understand you're having connectivity issues. "
            "We'd be happy to look into this with you. "
            "Please send us a DM with more details so we can "
            "troubleshoot the issue further."
        )


    # --------------------------------------------------------
    # App issue
    # --------------------------------------------------------

    if intent == "app_issue":

        return (
            "We're sorry you're having trouble with the app. "
            "We'd be happy to help troubleshoot the issue. "
            "Please send us a DM with more details so we can "
            "look into it further."
        )


    # --------------------------------------------------------
    # Hardware
    # --------------------------------------------------------

    if intent == "hardware_device":

        return (
            "We're sorry you're experiencing an issue with "
            "your device. We'd be happy to look into this "
            "with you. Please send us a DM with more details "
            "so we can troubleshoot the issue."
        )


    # --------------------------------------------------------
    # Software update
    # --------------------------------------------------------

    if intent == "software_update":

        return (
            "We understand you're having trouble with the "
            "software update. We'd be happy to look into "
            "what's happening. Please send us a DM with "
            "more details so we can troubleshoot further."
        )


    # --------------------------------------------------------
    # Services / media
    # --------------------------------------------------------

    if intent == "services_media":

        return (
            "We're sorry you're having trouble with the "
            "service. We'd be happy to look into this with "
            "you. Please send us a DM with more details so "
            "we can troubleshoot the issue."
        )


    # --------------------------------------------------------
    # Account access
    # --------------------------------------------------------

    if intent == "account_access":

        return (
            "We understand you're having trouble accessing "
            "your account. Please send us a DM with more "
            "details so our support team can securely look "
            "into the issue with you."
        )


    # --------------------------------------------------------
    # Billing / payment
    # --------------------------------------------------------

    if intent == "billing_payment":

        return (
            "We understand your concern about the billing "
            "or payment issue. Please send us a DM with "
            "more details so our support team can securely "
            "look into this with you."
        )


    # --------------------------------------------------------
    # Order / purchase
    # --------------------------------------------------------

    if intent == "order_purchase":

        return (
            "We understand your concern about your order. "
            "Please send us a DM with the relevant details "
            "so our support team can look into the issue "
            "with you."
        )


    # --------------------------------------------------------
    # General troubleshooting
    # --------------------------------------------------------

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
    intent_margin,
    retrieved_cases,
):
    """
    Decide whether the support agent should automatically
    handle the request or escalate it to a human.

    Rules:
    - Sensitive intents -> human
    - No historical evidence -> human
    - Low intent confidence -> human
    - Low top-1 vs top-2 margin -> human
    - Strong historical similarity -> auto-handle
    - Otherwise -> human
    """

    # --------------------------------------------------------
    # Sensitive cases
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # No historical evidence
    # --------------------------------------------------------

    if not retrieved_cases:

        return (
            "ESCALATE_TO_HUMAN",
            "No relevant historical support cases found.",
        )


    # --------------------------------------------------------
    # Low semantic intent confidence
    # --------------------------------------------------------

    if confidence < 0.30:

        return (
            "ESCALATE_TO_HUMAN",
            "Intent classification confidence is low.",
        )


    # --------------------------------------------------------
    # Ambiguous intent
    # --------------------------------------------------------

    if intent_margin < 0.05:

        return (
            "ESCALATE_TO_HUMAN",
            "Top intents are too close, making "
            "classification ambiguous.",
        )


    # --------------------------------------------------------
    # Top retrieval similarity
    # --------------------------------------------------------

    top_similarity = float(
        retrieved_cases[0]["similarity"]
    )


    # --------------------------------------------------------
    # Strong historical evidence
    # --------------------------------------------------------

    if top_similarity >= 0.75:

        return (
            "AUTO_HANDLE",
            "Intent is sufficiently clear and there is "
            "strong similarity with relevant historical "
            "support cases.",
        )


    # --------------------------------------------------------
    # Default escalation
    # --------------------------------------------------------

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
    exclude_tweet_ids=None,
    exclude_customer_texts=None,
):
    """
    Run the complete support-agent pipeline.

    Pipeline:
        Customer message
            ->
        Intent classification
            ->
        Similar-case retrieval
            ->
        Grounded response
            ->
        Auto-handle / escalation decision
    """

    # --------------------------------------------------------
    # 1. Semantic intent classification
    # --------------------------------------------------------

    (
        intent,
        confidence,
        ranked_intents,
        second_intent,
        second_score,
        intent_margin,
    ) = classify_intent(
        customer_message
    )


    # --------------------------------------------------------
    # 2. Intent-aware retrieval
    # --------------------------------------------------------

    retrieved_cases = retrieve_similar_cases(
        customer_message,
        intent,
        top_k=3,
        exclude_tweet_ids=exclude_tweet_ids,
        exclude_customer_texts=exclude_customer_texts,
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
        intent_margin,
        retrieved_cases,
    )


    # --------------------------------------------------------
    # 5. Return complete result
    # --------------------------------------------------------

    return {
        "intent": intent,
        "confidence": confidence,
        "second_intent": second_intent,
        "second_score": second_score,
        "intent_margin": intent_margin,
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

    print(
        "AI SUPPORT AGENT RESULT"
    )

    print("=" * 70)


    print(
        f"\nIntent          : "
        f"{result['intent']}"
    )


    print(
        f"Confidence      : "
        f"{result['confidence']:.4f}"
    )


    print(
        f"Second intent   : "
        f"{result['second_intent']}"
    )


    print(
        f"Second score    : "
        f"{result['second_score']:.4f}"
    )


    print(
        f"Intent margin   : "
        f"{result['intent_margin']:.4f}"
    )


    print(
        f"Decision        : "
        f"{result['decision']}"
    )


    print(
        f"Reason          : "
        f"{result['reason']}"
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

    print(
        "REFERENCE CASES"
    )

    print("=" * 70)


    for i, row in enumerate(
        result["retrieved_cases"],
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