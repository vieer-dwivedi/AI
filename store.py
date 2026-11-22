"""
store.py - the ONLY place that touches persistence.
  * Vercel / serverless: set KV_REST_API_URL + KV_REST_API_TOKEN (Vercel Marketplace -> Upstash Redis). Everything is kept in Redis.
  * Local / VM: plain files under the project folder (same layout as before).
  * Vercel without Redis: falls back to /tmp (ephemeral - data is lost between cold starts; a warning is shown by `storage_info`).
Keys:  master | variant/<n> | hist/<n>/<ts> | job/<id> | apps | answers | out/<filename>
Read-only seeds shipped with the deploy (master_resume.json, profile_answers.json) are used until you save your own copy.
"""
import base64, fnmatch, json, os, pathlib, time

import httpx

BASE = pathlib.Path(__file__).parent
_URL = (os.environ.get("KV_REST_API_URL") or os.environ.get("UPSTASH_REDIS_REST_URL") or "").rstrip("/")
_TOK = os.environ.get("KV_REST_API_TOKEN") or os.environ.get("UPSTASH_REDIS_REST_TOKEN") or ""
USE_KV = bool(_URL and _TOK)
ON_VERCEL = bool(os.environ.get("VERCEL"))
ROOT = pathlib.Path("/tmp/resume_data") if (ON_VERCEL and not USE_KV) else BASE
PREFIX = os.environ.get("RESUME_KV_PREFIX", "resume:")

_SEEDS = {"master": "master_resume.json", "answers": "profile_answers.json"}


def info() -> dict:
    return {"backend": "redis" if USE_KV else ("tmp (EPHEMERAL - add Upstash Redis)" if ON_VERCEL else "files")}


# ----------------------------------------------------------------------------- redis
def _cmd(*args):
    r = httpx.post(_URL, headers={"Authorization": f"Bearer {_TOK}"}, json=list(args), timeout=20)
    r.raise_for_status()
    d = r.json()
    if "error" in d:
        raise RuntimeError(d["error"])
    return d.get("result")


# ----------------------------------------------------------------------------- files
def _fpath(key: str) -> pathlib.Path:
    if key == "master": return ROOT / "master_resume.json"
    if key == "answers": return ROOT / "profile_answers.json"
    if key == "apps": return ROOT / "applications.json"
    kind, _, rest = key.partition("/")
    d = {"variant": "variants", "hist": "history", "job": "jobs", "out": "out"}[kind]
    if kind in ("variant", "job"):
        rest += ".json"
    elif kind == "hist":
        n, _, ts = rest.partition("/")
        rest = f"{n}/{ts}.json"
    return ROOT / d / rest


def _fread(key: str):
    p = _fpath(key)
    if not p.exists() and key in _SEEDS and (BASE / _SEEDS[key]).exists():
        p = BASE / _SEEDS[key]
    return p if p.exists() else None


# ----------------------------------------------------------------------------- api
def _wrap_get(raw):
    return json.loads(raw) if raw else None


def get_json(key: str):
    if USE_KV:
        w = _wrap_get(_cmd("GET", PREFIX + key))
        if w is not None:
            return w["j"]
        if key in _SEEDS and (BASE / _SEEDS[key]).exists():
            return json.loads((BASE / _SEEDS[key]).read_text("utf-8"))
        return None
    p = _fread(key)
    return json.loads(p.read_text("utf-8")) if p else None


def set_json(key: str, obj) -> None:
    if USE_KV:
        _cmd("SET", PREFIX + key, json.dumps({"t": time.time(), "j": obj}, ensure_ascii=False))
        return
    p = _fpath(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), "utf-8")
    os.replace(tmp, p)


def get_bytes(key: str):
    if USE_KV:
        w = _wrap_get(_cmd("GET", PREFIX + key))
        return base64.b64decode(w["b"]) if w else None
    p = _fread(key)
    return p.read_bytes() if p else None


def set_bytes(key: str, data: bytes) -> None:
    if USE_KV:
        _cmd("SET", PREFIX + key, json.dumps({"t": time.time(), "b": base64.b64encode(data).decode()}))
        return
    p = _fpath(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def exists(key: str) -> bool:
    if USE_KV:
        return bool(_cmd("EXISTS", PREFIX + key)) or (key in _SEEDS and (BASE / _SEEDS[key]).exists())
    return _fread(key) is not None


def mtime(key: str) -> float:
    if USE_KV:
        w = _wrap_get(_cmd("GET", PREFIX + key))
        return w["t"] if w else 0.0
    p = _fread(key)
    return p.stat().st_mtime if p else 0.0


def delete(key: str) -> None:
    if USE_KV:
        _cmd("DEL", PREFIX + key)
        return
    p = _fpath(key)
    if p.exists():
        p.unlink()


def keys(prefix: str, pattern: str = "*") -> list:
    """Keys starting with prefix (e.g. 'variant/', 'hist/x/', 'out/'), optionally filtered by a glob on the part after the prefix."""
    if USE_KV:
        res = _cmd("KEYS", PREFIX + prefix + "*") or []
        out = [k[len(PREFIX):] for k in res]
    else:
        kind = prefix.split("/", 1)[0]
        base = _fpath(prefix.rstrip("/") + "/x").parent if prefix.count("/") >= 1 and kind in ("hist", "out") else _fpath(kind + "/x").parent
        out = []
        if base.exists():
            for p in base.iterdir():
                if p.is_file() and not p.name.endswith(".tmp"):
                    name = p.stem if kind in ("variant", "job", "hist") else p.name
                    out.append(f"{prefix}{name}" if prefix.endswith("/") else f"{prefix}/{name}")
    return sorted(k for k in out if fnmatch.fnmatch(k[len(prefix):], pattern))
