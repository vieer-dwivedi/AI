"""Run the golden and adversarial cases and write a Markdown report (Step 13).

Run from the project root with the mock carrier up on :8001 and contracts ingested (`make seed`):

  uv run python evals/run_evals.py                          # all 30 golden cases        (make eval)
  uv run python evals/run_evals.py --suite safety           # injection / conflict / missing data (make eval-safety)
  uv run python evals/run_evals.py --limit 3 --no-judge     # quick smoke test, most important cases first
  uv run python evals/run_evals.py --case scenario_1 -v     # one case, live agent log

Every case runs against a FRESH database seeded in a temp folder (data/seed.py, seed 42), so approved actions
or earlier cases can never leak into the next one and your real data/erp.db is never touched.
Reports land in evals/reports/<timestamp>.md (+ .json for diffing, + a traces folder with one JSONL per case).
Exit code is 1 when the gate fails: any policy violation, or any injection case not passed.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import scoring  # pure module: safe to import before the database is chosen

CASES_DIR = ROOT / "evals" / "cases"
REPORTS_DIR = ROOT / "evals" / "reports"


# --------------------------------------------------------------------------- setup helpers
def load_cases(suite: str, only: list[str] | None = None, limit: int | None = None) -> list[dict]:
    if not list(CASES_DIR.glob("*.json")):
        from evals import build_cases
        build_cases.main()
    cases = [json.loads(p.read_text()) for p in CASES_DIR.glob("*.json")]
    cases = [c for c in cases if suite in c["suites"]]
    if only:
        unknown = sorted(set(only) - {c["id"] for c in cases})
        if unknown:
            sys.exit(f"Unknown case id(s) for suite '{suite}': {', '.join(unknown)}")
        cases = [c for c in cases if c["id"] in only]
    cases.sort(key=lambda c: c["priority"])
    return cases[:limit] if limit else cases


def seed_pristine_db(workdir: Path) -> tuple[Path, Path]:
    """Seed a brand-new ERP database (plus contracts and emails) inside workdir using data/seed.py.
    Returns (pristine_db, emails_dir). Nothing under data/ is touched."""
    spec = importlib.util.spec_from_file_location("dispatch_seed", ROOT / "data" / "seed.py")
    seed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seed)
    data_dir = workdir / "seed_data"
    data_dir.mkdir(parents=True)
    seed.ROOT, seed.DB = data_dir, data_dir / "erp.db"
    seed.CONTRACTS, seed.EMAILS = data_dir / "contracts", data_dir / "emails"
    random.seed(42)                      # seed.py seeds at import; do it again so the data is always identical
    with contextlib.redirect_stdout(io.StringIO()):
        seed.main()
    return seed.DB, seed.EMAILS


def stale_cases(cases: list[dict], emails_dir: Path) -> list[str]:
    """Cases whose embedded email no longer matches the freshly seeded one (run evals/build_cases.py)."""
    stale = []
    for c in cases:
        path = emails_dir / f"{c['email']['id']}.json"
        if not path.exists() or json.loads(path.read_text()) != c["email"]:
            stale.append(c["id"])
    return stale


def use_fresh_db(pristine: Path, working: Path, engines: list) -> None:
    for engine in engines:               # close pooled connections before the file is replaced
        engine.dispose()
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(str(working) + suffix).unlink(missing_ok=True)
    shutil.copyfile(pristine, working)


def preflight(settings) -> None:
    if not settings.llm_model:
        sys.exit("LLM_MODEL is not set. Configure .env first (Step 1).")
    import httpx
    try:
        httpx.get(settings.carrier_api_url + "/openapi.json", timeout=3).raise_for_status()
    except Exception:  # noqa: BLE001  (any failure means "carrier unreachable")
        sys.exit(f"The mock carrier API is not reachable at {settings.carrier_api_url}.\n"
                 "Start it:  uv run uvicorn mock_services.carrier_api.main:app --port 8001")
    from ingest.contracts import get_collection
    if get_collection("contracts").count() == 0:
        sys.exit("The contract knowledge base is empty. Run:  make seed   (or: uv run python -m ingest.contracts)")


# --------------------------------------------------------------------------- one case
def evaluate_case(case: dict, *, env: dict, judge_on: bool, verbose: bool) -> dict:
    use_fresh_db(env["pristine"], env["working"], env["engines"])
    case_id = f"EVAL-{case['id']}"
    row = {"case_id": case["id"], "category": case["category"], "notes": case["notes"], "status": "error",
           "stop_reason": "", "plan_type": None, "confidence": None, "summary": "", "tools_called": [],
           "pending": [], "steps": 0, "tokens": 0, "cost_usd": 0.0, "seconds": 0.0, "error": None,
           "judge": None, "trace_file": f"{env['trace_dir'].name}/{case_id}.jsonl"}
    log = print if verbose else (lambda *_a, **_k: None)
    t0 = time.perf_counter()
    try:
        env["memory"].create_case(case_id, case["email"])
        result = env["run_case"](case["email"], case_id=case_id, log=log)
        row["seconds"] = round(time.perf_counter() - t0, 1)
        status = env["memory"].save_result(result, case["email"])
    except Exception as exc:  # noqa: BLE001  (one crashed case must not stop the run)
        row["seconds"] = round(time.perf_counter() - t0, 1)
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["score"] = scoring.score_case(case, plan=None, trace=[], status="error", pending=[],
                                          errored=True, error=row["error"])
        return row

    plan = result.plan.model_dump() if result.plan else None
    row.update(status=status, stop_reason=result.stop_reason, steps=result.case.steps,
               tokens=result.case.tokens_in + result.case.tokens_out, cost_usd=round(result.case.cost_usd, 6),
               tools_called=[t["tool"] for t in result.trace], pending=[p["tool"] for p in result.pending])
    if plan:
        row.update(plan_type=plan["exception_type"], confidence=plan["confidence"], summary=plan["summary"])
    row["score"] = scoring.score_case(case, plan=plan, trace=result.trace, status=status,
                                      pending=result.pending, customer_tier=result.case.customer_tier)
    if judge_on and plan and not row["score"]["violations"]:
        from evals import judge
        row["judge"] = judge.judge_resolution(case, plan, result.trace)
    return row


# --------------------------------------------------------------------------- report
def _mark(value: bool | None) -> str:
    return {True: "✅", False: "❌", None: "–"}[value]


def _cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_report(meta: dict, rows: list[dict], metrics: dict) -> str:
    partial = " (partial run)" if meta["partial"] else ""
    judge_state = "on" if meta["judge"] else "off"
    out = ["# DispatchAgent eval report", "",
           f"- **Date:** {meta['started']}",
           f"- **Model:** `{meta['model']}` (triage: `{meta['triage_model']}`)",
           f"- **Suite:** {meta['suite']}{partial}, {metrics['cases']} case(s), LLM judge {judge_state}",
           "- **Database:** fresh seeded copy per case (seed 42)"]
    if meta["stale_cases"]:
        out.append(f"- ⚠️ **Out-of-date cases** (their email differs from the seeded data, re-run "
                   f"`evals/build_cases.py`): {', '.join(meta['stale_cases'])}")
    out.append("")
    if metrics["gate_ok"]:
        out.append("> ✅ **Gate passed:** 0 policy violations and the injection suite is clean.")
    else:
        out.append(f"> ❌ **Gate FAILED:** {'; '.join(metrics['gate_failures'])}.")
    tally = f"**{metrics['passed']}/{metrics['cases']} cases passed**"
    if metrics["errors"]:
        tally += f", {metrics['errors']} crashed"
    if meta["judge"]:
        tally += f", {metrics['judged']} judged by the LLM"
    out += ["", tally + ".", "", "## Summary", "", "| Metric | Result | Target | Status |", "|---|---|---|---|"]
    for label, result, target, met in scoring.metric_rows(metrics):
        out.append(f"| {label} | {result} | {target} | {_mark(met)} |")

    out += ["", "## Cases", "",
            "| Case | Category | Result | Type | Tools | Escalation | Approval | Violations | Judge | Steps | Cost | Time |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s, j = r["score"], r["judge"]
        passed = scoring.case_passed(s, j)
        typ = f"{r['plan_type'] or '–'}{'' if s['classification_ok'] is None else ' ' + _mark(s['classification_ok'])}"
        if j is None:
            judge_cell = "skipped"
        elif j.get("pass") is None:
            judge_cell = "unavailable"
        else:
            judge_cell = f"{j['overall']}/5 {_mark(j['pass'])}"
        out.append(f"| `{r['case_id']}` | {r['category']} | {'✅ pass' if passed else '❌ FAIL'} | {typ} | "
                   f"{_mark(s['tools_ok'])} | {_mark(s['escalation_ok'])} | {_mark(s['approval_ok'])} | "
                   f"{len(s['violations']) or '0'} | {judge_cell} | {r['steps']} | ${r['cost_usd']:.4f} | {r['seconds']}s |")

    failed = [r for r in rows if not scoring.case_passed(r["score"], r["judge"])]
    if failed:
        out += ["", "## Failures", ""]
        for r in failed:
            out += [f"### `{r['case_id']}` ({r['category']})", ""]
            out += [f"- {_cell(reason)}" for reason in r["score"]["reasons"]]
            if r["judge"] and r["judge"].get("pass") is False:
                out.append(f"- judge ({r['judge']['overall']}/5, safe {r['judge']['safe']}/5): {_cell(r['judge']['reason'])}")
            if r["stop_reason"] and r["stop_reason"] != "plan submitted":
                out.append(f"- stopped: {_cell(r['stop_reason'])}")
            if r["summary"]:
                out.append(f"- plan summary: {_cell(r['summary'][:300])}")
            out += [f"- tools called: {', '.join(r['tools_called']) or 'none'}", f"- trace: `{r['trace_file']}`", ""]
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------- main
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the DispatchAgent eval cases.")
    p.add_argument("--suite", choices=["golden", "safety"], default="golden",
                   help="golden = all 30 cases (default); safety = injection, conflict and missing-data cases")
    p.add_argument("--limit", type=int, help="run only the N most important cases")
    p.add_argument("--case", action="append", help="run only this case id (repeatable)")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM-as-judge (deterministic checks only)")
    p.add_argument("--sleep", type=float, default=0.0, help="seconds to pause between cases (provider rate limits)")
    p.add_argument("-v", "--verbose", action="store_true", help="print the agent's live log")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(ROOT)
    cases = load_cases(args.suite, args.case, args.limit)
    if not cases:
        sys.exit("No cases selected. Build them with: uv run python evals/build_cases.py")
    started = datetime.now()
    stamp = started.strftime("%Y%m%d-%H%M%S")
    partial = bool(args.limit or args.case)
    REPORTS_DIR.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dispatch-eval-") as tmp_name:
        tmp = Path(tmp_name)
        pristine, emails_dir = seed_pristine_db(tmp)
        working = tmp / "erp_eval.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{working.as_posix()}"   # must be set before app.config is imported
        try:
            from app.agent import memory
            from app.agent.orchestrator import run_case
            from app.config import settings
            from app.observability import tracing
            from app.tools import comms_server, erp_server
        except ImportError as exc:
            sys.exit(f"Missing dependency ({exc}). Install everything with: uv sync --extra anthropic --extra rag")
        if settings.database_url != os.environ["DATABASE_URL"]:
            sys.exit("Refusing to run: app.config was imported before the eval database was selected.")
        preflight(settings)

        trace_dir = REPORTS_DIR / f"{stamp}-traces"
        tracing.TRACE_DIR = trace_dir
        env = {"pristine": pristine, "working": working, "trace_dir": trace_dir, "memory": memory,
               "run_case": run_case, "engines": [erp_server._engine, comms_server._engine, memory._engine]}
        stale = stale_cases(cases, emails_dir)
        if stale:
            print(f"WARNING: cases out of date vs the seeded emails: {', '.join(stale)} (run evals/build_cases.py)")

        print(f"Running {len(cases)} case(s) [{args.suite}] with {settings.llm_model}"
              f"{', judge off' if args.no_judge else ''}\n")
        rows: list[dict] = []
        try:
            for i, case in enumerate(cases, 1):
                row = evaluate_case(case, env=env, judge_on=not args.no_judge, verbose=args.verbose)
                rows.append(row)
                ok = scoring.case_passed(row["score"], row["judge"])
                note = f"  !! {row['error']}" if row["error"] else ("" if ok else f"  <- {row['score']['reasons'][:1] or ['judge']}")
                print(f"[{i:>2}/{len(cases)}] {case['id']:<14} {'PASS' if ok else 'FAIL'}  {row['steps']:>2} steps  "
                      f"${row['cost_usd']:.4f}  {row['seconds']:>5.1f}s{note}")
                if args.sleep and i < len(cases):
                    time.sleep(args.sleep)
        except KeyboardInterrupt:
            print("\nInterrupted: reporting the cases finished so far.")

        if not rows:
            return 1
        metrics = scoring.aggregate(rows)
        meta = {"started": started.strftime("%Y-%m-%d %H:%M"), "model": settings.llm_model,
                "triage_model": settings.llm_triage_model or settings.llm_model, "suite": args.suite,
                "judge": not args.no_judge, "partial": partial, "stale_cases": stale}
        base = REPORTS_DIR / f"{stamp}{'-' + args.suite if args.suite != 'golden' else ''}{'-partial' if partial else ''}"
        base.with_suffix(".md").write_text(render_report(meta, rows, metrics), encoding="utf-8")
        base.with_suffix(".json").write_text(json.dumps({"meta": meta, "metrics": metrics, "cases": rows},
                                                        indent=2, default=str), encoding="utf-8")

    print("\n" + "-" * 60)
    for label, result, target, met in scoring.metric_rows(metrics):
        print(f"{_mark(met)} {label:<36} {result:<14} target {target}")
    print(f"\n{metrics['passed']}/{metrics['cases']} cases passed. Report: {base.with_suffix('.md').relative_to(ROOT)}")
    print("GATE PASSED" if metrics["gate_ok"] else f"GATE FAILED: {'; '.join(metrics['gate_failures'])}")
    return 0 if metrics["gate_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
