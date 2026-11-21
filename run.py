#!/usr/bin/env python3
"""
One command to run everything:   python run.py

 - loads .env (creates MCP_SECRET automatically on first run)
 - starts the MCP server on 127.0.0.1:PORT
 - sets up / reuses a Cloudflare named tunnel and routes MCP_DOMAIN -> the server
 - prints the URL to paste into Claude (Settings -> Connectors -> Add custom connector)
"""
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ENV_FILE = HERE / ".env"
CF_DIR = HERE / ".cloudflared"


def load_env() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k.strip() and v != "":
            os.environ.setdefault(k.strip(), v)


def save_env_var(key: str, value: str) -> None:
    with open(ENV_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{key}={value}\n")


def die(msg: str) -> None:
    print(f"\n[error] {msg}\n")
    sys.exit(1)


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for_port(port: int, timeout: float = 20) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if port_open(port):
            return True
        time.sleep(0.3)
    return False


def cf_run(cf: str, *args: str, check: bool = True) -> str:
    r = subprocess.run([cf, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        die(f"cloudflared {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def prepare_named_tunnel(cf: str, name: str, domain: str, port: int) -> list:
    """Create (or reuse) a named tunnel, point DNS at it, write config, return the run command."""
    home_cf = Path.home() / ".cloudflared"
    if not (home_cf / "cert.pem").exists():
        print("[tunnel] First-time login: a browser window will open. Pick the domain's zone and authorize.")
        subprocess.run([cf, "tunnel", "login"], check=True)

    def find():
        out = cf_run(cf, "tunnel", "list", "-o", "json").strip()
        data = json.loads(out) if out and out != "null" else []
        return next((t for t in data if t.get("name") == name), None)

    t = find()
    if not t:
        print(f"[tunnel] Creating tunnel '{name}'...")
        cf_run(cf, "tunnel", "create", name)
        t = find()
        if not t:
            die("Tunnel was created but could not be found afterwards.")
    tid = t["id"]
    cred = home_cf / f"{tid}.json"
    if not cred.exists():
        die(f"Tunnel '{name}' exists but its credentials file is missing ({cred}).\n"
            f"Delete it with `cloudflared tunnel delete {name}` and re-run, or set a different TUNNEL_NAME.")

    print(f"[tunnel] Routing DNS {domain} -> tunnel '{name}'...")
    cf_run(cf, "tunnel", "route", "dns", "--overwrite-dns", name, domain)

    CF_DIR.mkdir(exist_ok=True)
    cfg = CF_DIR / "config.yml"
    cfg.write_text(
        f"tunnel: {tid}\n"
        f"credentials-file: \"{cred.as_posix()}\"\n"
        f"ingress:\n"
        f"  - hostname: {domain}\n"
        f"    service: http://127.0.0.1:{port}\n"
        f"    originRequest:\n"
        f"      httpHostHeader: 127.0.0.1:{port}\n"
        f"  - service: http_status:404\n",
        encoding="utf-8",
    )
    return [cf, "tunnel", "--no-autoupdate", "--config", str(cfg), "run", name]


def main() -> None:
    load_env()

    root = os.environ.get("MCP_ROOT", "").strip()
    if not root:
        die("Set MCP_ROOT in .env (the folder Claude is allowed to read/write). See .env.example.")
    if not Path(root).expanduser().is_dir():
        die(f"MCP_ROOT is not a directory: {root}")

    if not os.environ.get("MCP_SECRET"):
        s = secrets.token_urlsafe(32)
        save_env_var("MCP_SECRET", s)
        os.environ["MCP_SECRET"] = s
        print("[setup] Generated MCP_SECRET and saved it to .env")

    port = int(os.environ.get("MCP_PORT", "8787"))
    domain = os.environ.get("MCP_DOMAIN", "").strip().replace("https://", "").strip("/")
    token = os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", "").strip()
    tunnel_name = os.environ.get("TUNNEL_NAME", "file-mcp")
    skip_tunnel = os.environ.get("SKIP_TUNNEL", "0").lower() in ("1", "true", "yes")

    server_cmd = [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1",
                  "--port", str(port), "--log-level", "warning"]
    tunnel_cmd = None

    if not skip_tunnel:
        if not domain:
            die("Set MCP_DOMAIN in .env (e.g. mcp.yourdomain.com), or SKIP_TUNNEL=1 for local-only testing.")
        cf = shutil.which("cloudflared")
        if not cf:
            die("cloudflared is not installed. Install it: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/")
        if token:
            print("[tunnel] Using CLOUDFLARE_TUNNEL_TOKEN (public hostname must be configured in the Cloudflare dashboard -> "
                  f"service http://localhost:{port}).")
            tunnel_cmd = [cf, "tunnel", "--no-autoupdate", "run", "--token", token]
        else:
            tunnel_cmd = prepare_named_tunnel(cf, tunnel_name, domain, port)

    if port_open(port):
        die(f"Port {port} is already in use. Change MCP_PORT in .env or stop the other process.")

    procs: dict = {}

    def start(name: str, cmd: list):
        procs[name] = subprocess.Popen(cmd, cwd=HERE, env=os.environ.copy())

    start("server", server_cmd)
    if not wait_for_port(port):
        die("Server did not start. Run `pip install -r requirements.txt` and check the output above.")
    if tunnel_cmd:
        start("tunnel", tunnel_cmd)

    secret = os.environ["MCP_SECRET"]
    base = f"https://{domain}" if tunnel_cmd else f"http://127.0.0.1:{port}"
    print("\n" + "=" * 70)
    print(" MCP server is running")
    print(f" Root folder : {root}")
    print(f" Read-only   : {os.environ.get('MCP_READ_ONLY', '0')}")
    print(f" Run commands : {'DISABLED' if os.environ.get('MCP_ENABLE_EXEC', '1') in ('0', 'false', 'no') or os.environ.get('MCP_READ_ONLY', '0') in ('1', 'true', 'yes') else 'enabled (bash on this machine)'}")
    print(f"\n Paste this URL into Claude (Settings -> Connectors -> Add custom connector):\n")
    print(f"   {base}/{secret}/mcp\n")
    print(" Keep this URL private: anyone who has it can use the file tools.")
    print(" Press Ctrl+C to stop.")
    print("=" * 70 + "\n")

    cmds = {"server": server_cmd, "tunnel": tunnel_cmd}
    try:
        while True:
            time.sleep(2)
            for name, p in list(procs.items()):
                if p.poll() is not None:
                    print(f"[warn] {name} exited (code {p.returncode}); restarting in 3s...")
                    time.sleep(3)
                    start(name, cmds[name])
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()