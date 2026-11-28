"""Token usage and estimated USD cost per case (Step 12)."""
from app.config import settings


def estimate_usd(tokens_in: int, tokens_out: int) -> float:
    """Estimated cost from the price table in settings (USD per 1M tokens). Free tiers: set both to 0."""
    return (tokens_in * settings.price_in_usd_per_mtok + tokens_out * settings.price_out_usd_per_mtok) / 1_000_000


def track_usage(case, ai) -> dict:
    """Add one model call's usage to the case totals and return that call's own numbers.
    Providers that report no usage count as 0."""
    usage = getattr(ai, "usage_metadata", None) or {}
    t_in, t_out = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    cost = estimate_usd(t_in, t_out)
    case.tokens_in += t_in
    case.tokens_out += t_out
    case.cost_usd += cost
    return {"tokens_in": t_in, "tokens_out": t_out, "cost_usd": round(cost, 6)}
