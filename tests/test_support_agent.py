import sys
from pathlib import Path

# Allow importing modules from src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_intent_classifier import classify_intent
from response_generator import run_support_agent


def test_battery_message_classification():
    message = "My iPhone battery is draining very quickly"

    intent, confidence, ranked, second_intent, second_score, margin = (
        classify_intent(message)
    )

    assert intent == "battery_charging"
    assert confidence > 0
    assert second_intent != intent
    assert margin >= 0


def test_agent_returns_required_fields():
    message = "My iPhone battery is draining very quickly"

    result = run_support_agent(message)

    required_fields = [
        "intent",
        "confidence",
        "second_intent",
        "second_score",
        "intent_margin",
        "response",
        "decision",
        "reason",
        "retrieved_cases",
        "ranked_intents",
    ]

    for field in required_fields:
        assert field in result


def test_ambiguous_case_can_escalate():
    message = "My screen keeps freezing after the update"

    result = run_support_agent(message)

    assert result["decision"] in [
        "AUTO_HANDLE",
        "ESCALATE_TO_HUMAN",
    ]

    assert result["intent_margin"] >= 0


def test_response_is_not_empty():
    message = "My iPhone battery is draining very quickly"

    result = run_support_agent(message)

    assert isinstance(result["response"], str)
    assert len(result["response"].strip()) > 0