"""Pydantic models: ResolutionPlan, Option, PlannedAction, Citation, CaseState (Step 8)."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CaseState(BaseModel):
    """Minimal case facts the policy engine needs. Step 10 persists this."""
    case_id: str = "CASE-0"
    order_id: int | None = None
    customer_tier: int | None = None   # None = unknown, policy fails closed
    steps: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class Citation(BaseModel):
    source: Literal["tool", "contract", "playbook"]
    ref: str                           # tool call id, or e.g. "CON-007 §7.3"
    detail: str = ""


class Option(BaseModel):
    name: str
    description: str
    cost_inr: float = 0.0
    sla_penalty_inr: float = 0.0
    eta: str = ""
    risks: list[str] = Field(default_factory=list)


class PlannedAction(BaseModel):
    tool: str
    args: dict
    risk_tier: Literal["low", "medium", "high"]
    rationale: str = ""


class ResolutionPlan(BaseModel):
    case_id: str
    exception_type: Literal["delay", "damage", "stockout", "customs", "dispute", "other"]
    summary: str
    options: list[Option] = Field(min_length=1)
    recommended_option: int
    actions: list[PlannedAction] = Field(default_factory=list)
    evidence: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    needs_human_reason: str | None = None

    @model_validator(mode="after")
    def _check_plan(self):
        if not 0 <= self.recommended_option < len(self.options):
            raise ValueError("recommended_option must be a valid index into options")
        if self.confidence < 0.6 and not self.needs_human_reason:
            raise ValueError("confidence below 0.6 requires needs_human_reason")
        return self