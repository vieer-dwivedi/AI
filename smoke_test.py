"""End-to-end smoke test, no network, no Gmail. Runs the apply flow twice: file backend (tmp dir) and fake-Redis backend.
   python smoke_test.py"""
import asyncio, fnmatch, inspect, json, os, pathlib, sys, tempfile, time

os.environ.setdefault("RESUME_SECRET", "x" * 24)
os.environ["PUBLIC_BASE_URL"] = "https://example.vercel.app"
for k in ("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD"):
    os.environ.pop(k, None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import store, server as s

JD = """Azure Cloud and DevOps Engineer - Acme
Send your CV to careers@acme-test.com. We need 4+ years with Azure, Terraform, Kubernetes, Docker, Azure DevOps CI/CD pipelines,
monitoring with Prometheus and Grafana, Ansible and Python scripting. Nice to have: Helm, Argo CD."""


def call(_tool, **kw):
    fn = s.mcp._tool_manager._tools[_tool].fn
    r = fn(**kw)
    return asyncio.run(r) if inspect.iscoroutine(r) else r


def fake_redis():
    db = {}
    def cmd(*a):
        op = a[0]
        if op == "GET": return db.get(a[1])
        if op == "SET": db[a[1]] = a[2]; return "OK"
        if op == "DEL": return int(db.pop(a[1], None) is not None)
        if op == "EXISTS": return int(a[1] in db)
        if op == "KEYS": return [k for k in db if fnmatch.fnmatch(k, a[1])]
        raise RuntimeError(op)
    store._cmd, store.USE_KV = cmd, True


def run(label):
    t = time.time()
    st = call("apply_start", jd_text=JD)
    assert st["route"] == "email" and st["emails"] == ["careers@acme-test.com"], st
    jid, steps = st["job_id"], []
    for _ in range(20):
        n = call("apply_next", job_id=jid)
        steps.append(n["step"])
        if n["step"] == "done": break
        if n["step"] == "1_create_variant": call("create_variant", name=jid)
        elif n["step"] == "2_skills": call("optimize_skills_for_jd", variant=jid, jd_text=JD)
        elif n["step"] == "3_experience": call("sync_experience_for_jd", variant=jid, jd_text=JD)
        elif n["step"] == "4_headline_summary":
            call("set_basics", variant=jid, headline="Azure Cloud & DevOps Engineer",
                 summary="DevOps Engineer with 4+ years of experience automating cloud infrastructure, CI/CD and Kubernetes platforms on Azure and AWS using Terraform and Python.")
        elif n["step"] == "4_fix_validation": raise SystemExit(f"[{label}] validation errors: {n['errors']}")
        elif n["step"] == "5_render_pdf":
            r = call("render_pdf", variant=jid); assert r["ok"] and r["pages"] <= 3, r
            print(f"  pdf {r['pages']} pages @ {r['font_pt']}pt, missing={r['text_check_missing']}")
        elif n["step"] == "6_compose_email":
            a = n["do"]["args"]
            r = call("compose_email", variant=jid, role="Azure Cloud and DevOps Engineer", company="Acme", to=a["to"], pdf_path=a["pdf_path"],
                     hook="I am excited to apply for the Azure Cloud and DevOps Engineer role.",
                     key_points=["Automated cloud infrastructure with Terraform and CI/CD pipelines.", "Operated Kubernetes platforms on AWS and Azure."])
            assert r["attachment_included"], r
            if not r["ok"]: print("  email issues:", r["issues"])
        elif n["step"] == "7_send_with_your_gmail":
            assert "/dl/" in n["body_with_resume_link"], n
            call("mark_application_sent", job_id=jid, to=n["to"])
        else: raise SystemExit(f"[{label}] unexpected step {n['step']}")
    assert steps[-1] == "done", steps
    pdf = call("get_resume_pdf", job_id=jid)
    assert call("get_resume_pdf", job_id=jid, include_base64=True)["base64"][:4] == "JVBE"
    print(f"[{label}] OK in {time.time() - t:.1f}s  steps={steps}\n  {pdf['public_url']}")
    return jid


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        store.ROOT = pathlib.Path(d)
        run("files")
    fake_redis()
    run("redis")
