#!/usr/bin/env python3
"""Local dev only (Vercel runs api/index.py).   python run.py   ->  http://127.0.0.1:8788/<RESUME_SECRET>/mcp"""
import os, secrets, sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ENV = HERE / ".env"


def load_env():
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")


if __name__ == "__main__":
    load_env()
    if not os.environ.get("RESUME_SECRET"):
        s = secrets.token_urlsafe(32)
        with open(ENV, "a", encoding="utf-8") as f:
            f.write(f"\nRESUME_SECRET={s}\n")
        os.environ["RESUME_SECRET"] = s
    port = int(os.environ.get("RESUME_PORT", "8788"))
    os.environ["PORT"] = str(port)
    print(f"\nLocal: http://127.0.0.1:{port}/{os.environ['RESUME_SECRET']}/mcp\n(For Claude/ChatGPT use the Vercel URL.)\n", flush=True)
    import uvicorn
    sys.path.insert(0, str(HERE))
    uvicorn.run("server:app", host="127.0.0.1", port=port, log_level="warning")
