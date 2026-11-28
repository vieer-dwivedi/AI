import pytest
from pydantic import ValidationError

from app.agent.policy import Decision, evaluate
from app.agent.schemas import CaseState, ResolutionPlan

CASE = CaseState(customer_tier=2)


def test_unknown_tool_denied():
    assert evaluate("delete_database", {}, CASE) == Decision.DENY


def test_low_risk_allowed():
    assert evaluate("draft_customer_email", {}, CASE) == Decision.ALLOW
    assert evaluate("add_case_note", {}, CASE) == Decision.ALLOW


def test_update_order_status_allowed():
    assert evaluate("update_order_status", {"status": "delayed"}, CASE) == Decision.ALLOW


def test_expedite_over_threshold_needs_approval():
    assert evaluate("book_expedited_freight", {"cost_inr": 60000}, CASE) == Decision.NEEDS_APPROVAL


def test_expedite_at_threshold_allowed():
    assert evaluate("book_expedited_freight", {"cost_inr": 50000}, CASE) == Decision.ALLOW


def test_expedite_missing_cost_fails_closed():
    assert evaluate("book_expedited_freight", {}, CASE) == Decision.NEEDS_APPROVAL


def test_expedite_non_numeric_cost_fails_closed():
    assert evaluate("book_expedited_freight", {"cost_inr": "lots"}, CASE) == Decision.NEEDS_APPROVAL


def test_credit_over_threshold_needs_approval():
    assert evaluate("issue_credit_note", {"amount_inr": 30000}, CASE) == Decision.NEEDS_APPROVAL


def test_credit_at_threshold_allowed():
    assert evaluate("issue_credit_note", {"amount_inr": 25000}, CASE) == Decision.ALLOW


def test_reship_always_needs_approval():
    assert evaluate("create_reship_order", {"qty": 1}, CASE) == Decision.NEEDS_APPROVAL


def test_send_email_tier1_needs_approval():
    assert evaluate("send_email", {"draft_id": 1}, CaseState(customer_tier=1)) == Decision.NEEDS_APPROVAL


def test_send_email_tier2_allowed():
    assert evaluate("send_email", {"draft_id": 1}, CaseState(customer_tier=2)) == Decision.ALLOW


def test_send_email_unknown_tier_fails_closed():
    assert evaluate("send_email", {"draft_id": 1}, CaseState(customer_tier=None)) == Decision.NEEDS_APPROVAL


def test_step_budget_exhausted_denied():
    assert evaluate("add_case_note", {}, CaseState(customer_tier=2, steps=15)) == Decision.DENY


def test_cost_budget_exhausted_denied():
    assert evaluate("add_case_note", {}, CaseState(customer_tier=2, cost_usd=0.80)) == Decision.DENY


def test_malicious_rule_expression_fails_closed():
    policy = {"tools": {"x": {"require_approval_if": "__import__('os').system('echo hacked')"}}, "global": {}}
    assert evaluate("x", {}, CASE, policy) == Decision.NEEDS_APPROVAL


def _plan(**over):
    base = dict(case_id="C1", exception_type="delay", summary="s",
                options=[{"name": "air", "description": "air freight"}],
                recommended_option=0, confidence=0.9)
    return ResolutionPlan(**{**base, **over})


def test_valid_plan():
    assert _plan().recommended_option == 0


def test_low_confidence_requires_reason():
    with pytest.raises(ValidationError):
        _plan(confidence=0.4)
    assert _plan(confidence=0.4, needs_human_reason="missing tracking data").confidence == 0.4


def test_bad_recommended_index_rejected():
    with pytest.raises(ValidationError):
        _plan(recommended_option=5)