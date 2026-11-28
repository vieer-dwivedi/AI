"""Streamlit ops console: Inbox, Case, Approvals, Trace (Step 11).

Run: uv run streamlit run ui/console.py   (the API must be running on port 8000)
"""
import json
import os
import time
from pathlib import Path

import httpx
import streamlit as st

EMAILS = Path(__file__).resolve().parent.parent / "data" / "emails"

st.set_page_config(page_title="DispatchAgent Console", layout="wide")


def api(method: str, path: str, **kwargs):
    """Call the API; show an error and return None if it fails."""
    try:
        r = httpx.request(method, f"{st.session_state.api_url}{path}", timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error on {method} {path}: {exc}")
        return None


# ---------- sidebar: connection + launch a demo scenario ----------
st.session_state.setdefault("api_url", os.getenv("DISPATCH_API_URL", "http://localhost:8000"))
with st.sidebar:
    st.title("DispatchAgent")
    st.session_state.api_url = st.text_input("API URL", st.session_state.api_url)
    st.divider()
    n = st.selectbox("Demo scenario", [1, 2, 3, 4, 5, 6])
    if st.button("Run scenario", width="stretch"):
        email = json.loads((EMAILS / f"scenario_{n}.json").read_text())
        res = api("POST", "/exceptions", json=email)
        if res:
            st.session_state.selected_case = res["case_id"]   # set before the selector is drawn
    if st.button("Refresh", width="stretch"):
        st.rerun()

cases = api("GET", "/cases") or []
ids = [c["case_id"] for c in cases]
if not ids:
    st.info("No cases yet. Run a scenario from the sidebar.")
    st.stop()
if st.session_state.get("selected_case") not in ids:
    st.session_state.selected_case = ids[0]

st.selectbox("Selected case", ids, key="selected_case")
case = api("GET", f"/cases/{st.session_state.selected_case}") or {}

inbox, case_tab, approvals_tab, trace_tab = st.tabs(["Inbox", "Case", "Approvals", "Trace"])

# ---------- Inbox ----------
with inbox:
    st.dataframe(cases, width="stretch", hide_index=True)

# ---------- Case ----------
with case_tab:
    st.subheader(f"{case.get('case_id', '')}  ·  {case.get('status', '')}")
    email = case.get("email") or {}
    with st.expander("Original email", expanded=True):
        st.markdown(f"**{email.get('subject', '')}**")
        st.text(email.get("body", ""))

    plan = case.get("plan")
    if case.get("status") == "running":
        st.info("The agent is investigating...")
    elif case.get("status") == "failed":
        st.error(case.get("error", "failed"))
    elif not plan:
        st.warning(f"No valid plan ({case.get('stop_reason', '')}). A human needs to handle this case.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Type", plan["exception_type"])
        c2.metric("Confidence", plan["confidence"])
        c3.metric("Cost (USD)", (case.get("usage") or {}).get("cost_usd", 0))
        st.write(plan["summary"])
        if plan.get("needs_human_reason"):
            st.warning(f"Needs human: {plan['needs_human_reason']}")

        st.markdown("**Options: cost vs penalty**")
        rec = plan["recommended_option"]
        st.dataframe([{"recommended": "★" if i == rec else "", "option": o["name"], "cost_inr": o["cost_inr"],
                       "sla_penalty_inr": o["sla_penalty_inr"], "eta": o["eta"], "risks": ", ".join(o["risks"])}
                      for i, o in enumerate(plan["options"])], width="stretch", hide_index=True)

        st.markdown("**Planned actions**")
        st.dataframe([{"tool": a["tool"], "risk": a["risk_tier"], "args": json.dumps(a["args"]),
                       "why": a["rationale"]} for a in plan["actions"]], width="stretch", hide_index=True)

        st.markdown("**Evidence**")
        for e in plan["evidence"]:
            st.write(f"- `{e['source']}` {e['ref']}: {e['detail']}")

    if case.get("executed_actions"):
        st.markdown("**Executed after approval**")
        st.dataframe(case["executed_actions"], width="stretch", hide_index=True)

# ---------- Approvals ----------
with approvals_tab:
    pending = api("GET", "/approvals") or []
    if not pending:
        st.success("No pending approvals.")
    for a in pending:
        with st.expander(f"#{a['id']}  {a['tool_name']}  ({a['case_id']})", expanded=True):
            st.json(a["args"])
            comment = st.text_input("Comment (optional)", key=f"comment_{a['id']}")
            b1, b2 = st.columns(2)
            for col, label, ok in ((b1, "Approve", True), (b2, "Reject", False)):
                if col.button(label, key=f"{label}_{a['id']}", width="stretch"):
                    if api("POST", f"/approvals/{a['id']}", json={"approve": ok, "comment": comment}):
                        st.rerun()

    st.divider()
    st.subheader("Needs review")
    to_review = [c for c in cases if c["status"] in ("needs_review", "needs_human")]
    if not to_review:
        st.success("Nothing waiting for review.")
    for c in to_review:
        with st.expander(f"{c['case_id']}  ·  {c['exception_type'] or 'no plan'}  ·  confidence {c['confidence']}"):
            st.write(c["reason"] or "The agent asked for a human to look at this case.")
            note = st.text_input("Review note (optional)", key=f"note_{c['case_id']}")
            if st.button("Mark reviewed", key=f"rev_{c['case_id']}"):
                if api("POST", f"/cases/{c['case_id']}/review", json={"comment": note}):
                    st.rerun()

# ---------- Trace ----------
with trace_tab:
    events = api("GET", f"/cases/{st.session_state.selected_case}/trace") or []
    llm = [e for e in events if e["type"] == "llm_call"]
    if events:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model calls", len(llm))
        m2.metric("Tokens in", sum(e.get("tokens_in", 0) for e in llm))
        m3.metric("Tokens out", sum(e.get("tokens_out", 0) for e in llm))
        m4.metric("Model time (s)", round(sum(e.get("duration_s", 0) for e in llm), 1))
        st.markdown("**Timeline**")
        cols = ["ts", "type", "phase", "step", "tool", "decision", "tokens_in", "tokens_out", "duration_s"]
        st.dataframe([{k: e.get(k, "") for k in cols} for e in events], width="stretch", hide_index=True)
    else:
        st.info("No trace file for this case yet.")

    trace = case.get("trace") or []
    if trace:
        st.markdown("**Tool call details**")
    for t in trace:
        icon = {"allow": "✅", "needs_approval": "⏸️", "deny": "⛔"}.get(t["decision"], "•")
        with st.expander(f"Step {t['step']}  {icon} {t['tool']}  [{t['decision']}]"):
            st.caption("Arguments")
            st.json(t["args"])
            st.caption("Result")
            st.json(t["result"])

# while the agent is working, poll until it finishes
if case.get("status") == "running":
    time.sleep(3)
    st.rerun()
