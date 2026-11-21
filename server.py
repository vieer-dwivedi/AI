"""
File Tools MCP Server (Streamable HTTP)

Exposes file read / write / edit / search / move / delete tools, sandboxed to MCP_ROOT.
Run via run.py (recommended) or:  uvicorn server:app --host 127.0.0.1 --port 8787
"""
from __future__ import annotations

import asyncio
import atexit
import difflib
import fnmatch
import hmac
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

try:
    from mcp.server.transport_security import TransportSecuritySettings
except Exception:  # older SDKs
    TransportSecuritySettings = None

# --------------------------------------------------------------------------- config
ROOT = Path(os.environ.get("MCP_ROOT", str(Path.home()))).expanduser().resolve()
READ_ONLY = os.environ.get("MCP_READ_ONLY", "0").lower() in ("1", "true", "yes")
SECRET = os.environ.get("MCP_SECRET", "")
MAX_READ_BYTES = int(os.environ.get("MCP_MAX_READ_BYTES", str(1_000_000)))
MAX_WRITE_BYTES = int(os.environ.get("MCP_MAX_WRITE_BYTES", str(10_000_000)))
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".mypy_cache"}

if not ROOT.is_dir():
    raise SystemExit(f"MCP_ROOT does not exist or is not a directory: {ROOT}")

_kwargs = dict(stateless_http=True, json_response=True)
if TransportSecuritySettings is not None:
    # Host-header checks are for localhost-only servers; we are behind a tunnel + secret path.
    _kwargs["transport_security"] = TransportSecuritySettings(enable_dns_rebinding_protection=False)

try:
    mcp = FastMCP("file-tools", **_kwargs)
except TypeError:
    mcp = FastMCP("file-tools", stateless_http=True, json_response=True)

RO = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False)


# --------------------------------------------------------------------------- helpers
def _p(path: str | None) -> Path:
    """Resolve a path (relative to ROOT, or absolute inside ROOT) and enforce the sandbox."""
    if not path or path in (".", "/"):
        return ROOT
    raw = Path(path)
    full = raw if raw.is_absolute() else ROOT / raw
    resolved = full.resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError(f"Path is outside the allowed root ({ROOT}): {path}")
    return resolved


def _rel(p: Path) -> str:
    try:
        r = p.relative_to(ROOT).as_posix()
        return r or "."
    except ValueError:
        return str(p)


def _writable() -> None:
    if READ_ONLY:
        raise PermissionError("Server is in read-only mode (MCP_READ_ONLY=1).")


def _is_binary(p: Path) -> bool:
    with open(p, "rb") as f:
        return b"\x00" in f.read(8192)


def _read_text(p: Path) -> str:
    if not p.exists():
        raise FileNotFoundError(f"File not found: {_rel(p)}")
    if p.is_dir():
        raise IsADirectoryError(f"Is a directory: {_rel(p)}")
    if _is_binary(p):
        raise ValueError(f"Binary file, cannot read as text: {_rel(p)}")
    with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
        return f.read()


def _atomic_write(p: Path, content: str) -> int:
    data = content.encode("utf-8")
    if len(data) > MAX_WRITE_BYTES:
        raise ValueError(f"Content too large ({len(data)} bytes, limit {MAX_WRITE_BYTES}).")
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".mcp-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        if p.exists():
            try:
                shutil.copymode(p, tmp)
            except OSError:
                pass
        os.replace(tmp, p)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return len(data)


def _match_newlines(content: str, text: str) -> str:
    """If the file uses CRLF and the snippet uses LF, convert the snippet."""
    if "\r\n" in content and "\r\n" not in text:
        return text.replace("\n", "\r\n")
    return text


def _diff(old: str, new: str, name: str, max_lines: int = 60) -> str:
    d = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(), f"a/{name}", f"b/{name}", lineterm="", n=2))
    if len(d) > max_lines:
        d = d[:max_lines] + [f"... diff truncated ({len(d) - max_lines} more lines)"]
    return "\n".join(d)


def _walk(base: Path):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        yield Path(dirpath), sorted(filenames)


def _fmt_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


# --------------------------------------------------------------------------- read tools
@mcp.tool(annotations=RO)
def list_directory(path: str = ".", recursive: bool = False, max_depth: int = 3,
                   max_entries: int = 500) -> str:
    """List files and folders in a directory. Set recursive=true for a tree view (limited by max_depth)."""
    base = _p(path)
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {_rel(base)}")
    lines: list[str] = []

    def emit(d: Path, depth: int) -> bool:
        try:
            entries = sorted(d.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            lines.append("  " * depth + "[permission denied]")
            return True
        for e in entries:
            if len(lines) >= max_entries:
                return False
            if e.is_dir():
                lines.append("  " * depth + f"[DIR]  {e.name}/")
                if recursive and depth + 1 < max_depth and e.name not in SKIP_DIRS and not e.is_symlink():
                    if not emit(e, depth + 1):
                        return False
            else:
                try:
                    sz = _fmt_size(e.stat().st_size)
                except OSError:
                    sz = "?"
                lines.append("  " * depth + f"[FILE] {e.name} ({sz})")
        return True

    complete = emit(base, 0)
    header = f"Contents of {_rel(base)}:"
    if not lines:
        return header + "\n(empty)"
    out = header + "\n" + "\n".join(lines)
    if not complete:
        out += f"\n... truncated at {max_entries} entries"
    return out


@mcp.tool(annotations=RO)
def read_file(path: str, offset: int = 1, limit: int = 0, line_numbers: bool = False) -> str:
    """Read a text file. offset = first line (1-based), limit = max lines (0 = whole file).
    Set line_numbers=true to prefix lines with numbers (useful before replace_lines / insert_lines)."""
    p = _p(path)
    size = p.stat().st_size if p.exists() and p.is_file() else 0
    if size > MAX_READ_BYTES and limit <= 0:
        raise ValueError(
            f"File is {_fmt_size(size)} (limit {_fmt_size(MAX_READ_BYTES)}). Use offset and limit to read a range.")
    content = _read_text(p)
    lines = content.splitlines(keepends=True)
    total = len(lines)
    start = max(offset, 1) - 1
    end = total if limit <= 0 else min(start + limit, total)
    chunk = lines[start:end]
    if line_numbers:
        text = "".join(f"{start + i + 1}\t{l if l.endswith(chr(10)) else l + chr(10)}" for i, l in enumerate(chunk))
    else:
        text = "".join(chunk)
    if start > 0 or end < total:
        text = f"[lines {start + 1}-{end} of {total}]\n" + text
    return text


@mcp.tool(annotations=RO)
def read_multiple_files(paths: list[str]) -> str:
    """Read several text files at once. Failures are reported per file."""
    out = []
    for path in paths:
        try:
            p = _p(path)
            if p.is_file() and p.stat().st_size > MAX_READ_BYTES:
                raise ValueError("file too large, use read_file with offset/limit")
            out.append(f"=== {_rel(p)} ===\n{_read_text(p)}")
        except Exception as e:
            out.append(f"=== {path} ===\nERROR: {e}")
    return "\n\n".join(out)


@mcp.tool(annotations=RO)
def file_info(path: str) -> str:
    """Get metadata for a file or directory (type, size, timestamps, line count)."""
    p = _p(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {_rel(p)}")
    st = p.stat()
    info = [
        f"path: {_rel(p)}",
        f"type: {'directory' if p.is_dir() else 'file'}",
        f"size: {st.st_size} bytes ({_fmt_size(st.st_size)})",
        f"modified: {datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')}",
        f"created: {datetime.fromtimestamp(st.st_ctime).isoformat(timespec='seconds')}",
        f"permissions: {oct(st.st_mode & 0o777)}",
    ]
    if p.is_file() and not _is_binary(p) and st.st_size <= MAX_READ_BYTES:
        info.append(f"lines: {len(_read_text(p).splitlines())}")
    elif p.is_file():
        info.append("binary: " + str(_is_binary(p)).lower())
    return "\n".join(info)


@mcp.tool(annotations=RO)
def find_files(pattern: str, path: str = ".", max_results: int = 200) -> str:
    """Find files/folders by glob pattern on the name or relative path (e.g. '*.py', 'src/**/*.ts')."""
    base = _p(path)
    results: list[str] = []
    for d, files in _walk(base):
        for name in sorted(os.listdir(d)):
            full = d / name
            rel = _rel(full)
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern):
                results.append(rel + ("/" if full.is_dir() else ""))
                if len(results) >= max_results:
                    return "\n".join(results) + f"\n... truncated at {max_results} results"
    return "\n".join(results) if results else "No matches."


@mcp.tool(annotations=RO)
def search_text(query: str, path: str = ".", regex: bool = False, case_sensitive: bool = False,
                include: str = "*", context_lines: int = 0, max_results: int = 100) -> str:
    """Search file contents (like grep -rn). 'include' is a filename glob such as '*.py'."""
    base = _p(path)
    flags = 0 if case_sensitive else re.IGNORECASE
    pat = re.compile(query if regex else re.escape(query), flags)
    matches: list[str] = []
    files = [base] if base.is_file() else None

    def targets():
        if files:
            yield base
            return
        for d, names in _walk(base):
            for n in names:
                if fnmatch.fnmatch(n, include):
                    yield d / n

    for f in targets():
        try:
            if f.stat().st_size > MAX_READ_BYTES or _is_binary(f):
                continue
            lines = _read_text(f).splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines):
            if pat.search(line):
                if context_lines > 0:
                    a, b = max(0, i - context_lines), min(len(lines), i + context_lines + 1)
                    block = "\n".join(f"{_rel(f)}:{j + 1}{':' if j == i else '-'} {lines[j]}" for j in range(a, b))
                    matches.append(block + "\n--")
                else:
                    matches.append(f"{_rel(f)}:{i + 1}: {line.strip()[:300]}")
                if len(matches) >= max_results:
                    return "\n".join(matches) + f"\n... truncated at {max_results} results"
    return "\n".join(matches) if matches else "No matches."


# --------------------------------------------------------------------------- write tools
@mcp.tool(annotations=WRITE)
def create_file(path: str, content: str = "", overwrite: bool = False) -> str:
    """Create a new file (parent folders are created automatically). Fails if it exists unless overwrite=true."""
    _writable()
    p = _p(path)
    if p == ROOT:
        raise ValueError("Invalid path.")
    if p.is_dir():
        raise IsADirectoryError(f"A directory exists at {_rel(p)}")
    if p.exists() and not overwrite:
        raise FileExistsError(f"{_rel(p)} already exists. Use overwrite=true or edit_file.")
    n = _atomic_write(p, content)
    return f"Created {_rel(p)} ({n} bytes)"


@mcp.tool(annotations=DESTRUCTIVE)
def write_file(path: str, content: str) -> str:
    """Write a file, replacing its entire content if it exists (creates parent folders)."""
    _writable()
    p = _p(path)
    if p == ROOT or p.is_dir():
        raise IsADirectoryError(f"Cannot write to directory: {_rel(p)}")
    existed = p.exists()
    n = _atomic_write(p, content)
    return f"{'Overwrote' if existed else 'Created'} {_rel(p)} ({n} bytes)"


@mcp.tool(annotations=WRITE)
def append_file(path: str, content: str) -> str:
    """Append text to the end of a file (creates it if missing)."""
    _writable()
    p = _p(path)
    old = _read_text(p) if p.exists() else ""
    n = _atomic_write(p, old + content)
    return f"Appended to {_rel(p)} (now {n} bytes)"


@mcp.tool(annotations=DESTRUCTIVE)
def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace exact text in a file. old_string must match exactly once (include surrounding lines
    for uniqueness) unless replace_all=true. Returns a diff of the change."""
    _writable()
    if not old_string:
        raise ValueError("old_string must not be empty.")
    p = _p(path)
    content = _read_text(p)
    old_s, new_s = _match_newlines(content, old_string), _match_newlines(content, new_string)
    count = content.count(old_s)
    if count == 0:
        raise ValueError("old_string not found. Re-read the file and copy the text exactly (whitespace matters).")
    if count > 1 and not replace_all:
        raise ValueError(f"old_string matches {count} times. Add more context or set replace_all=true.")
    new_content = content.replace(old_s, new_s) if replace_all else content.replace(old_s, new_s, 1)
    _atomic_write(p, new_content)
    return f"Edited {_rel(p)} ({count if replace_all else 1} replacement(s))\n{_diff(content, new_content, _rel(p))}"


@mcp.tool(annotations=DESTRUCTIVE)
def edit_file_multi(path: str, edits: list[dict[str, str]]) -> str:
    """Apply several edits to one file atomically, in order. Each edit: {"old_string": ..., "new_string": ...}.
    Every old_string must match exactly once at the time it is applied; if any fails nothing is written."""
    _writable()
    p = _p(path)
    original = content = _read_text(p)
    for i, e in enumerate(edits, 1):
        if "old_string" not in e or "new_string" not in e or not e["old_string"]:
            raise ValueError(f"Edit {i}: needs non-empty old_string and new_string.")
        o, n = _match_newlines(content, e["old_string"]), _match_newlines(content, e["new_string"])
        c = content.count(o)
        if c != 1:
            raise ValueError(f"Edit {i}: old_string matched {c} times (need exactly 1). Nothing was written.")
        content = content.replace(o, n, 1)
    _atomic_write(p, content)
    return f"Applied {len(edits)} edit(s) to {_rel(p)}\n{_diff(original, content, _rel(p))}"


@mcp.tool(annotations=DESTRUCTIVE)
def replace_lines(path: str, start_line: int, end_line: int, new_text: str) -> str:
    """Replace lines start_line..end_line (1-based, inclusive) with new_text. Use '' to delete the lines."""
    _writable()
    p = _p(path)
    content = _read_text(p)
    lines = content.splitlines(keepends=True)
    if start_line < 1 or end_line < start_line or end_line > len(lines):
        raise ValueError(f"Invalid range {start_line}-{end_line}; file has {len(lines)} lines.")
    nl = "\r\n" if "\r\n" in content else "\n"
    repl = _match_newlines(content, new_text)
    if repl and not repl.endswith("\n") and end_line < len(lines):
        repl += nl
    new_content = "".join(lines[:start_line - 1]) + repl + "".join(lines[end_line:])
    _atomic_write(p, new_content)
    return f"Replaced lines {start_line}-{end_line} in {_rel(p)}\n{_diff(content, new_content, _rel(p))}"


@mcp.tool(annotations=WRITE)
def insert_lines(path: str, after_line: int, text: str) -> str:
    """Insert text after the given line number (0 = at the very beginning, or use the last line number to append)."""
    _writable()
    p = _p(path)
    content = _read_text(p)
    lines = content.splitlines(keepends=True)
    if after_line < 0 or after_line > len(lines):
        raise ValueError(f"after_line must be 0..{len(lines)}.")
    nl = "\r\n" if "\r\n" in content else "\n"
    ins = _match_newlines(content, text)
    if not ins.endswith("\n"):
        ins += nl
    if after_line > 0 and not lines[after_line - 1].endswith("\n"):
        lines[after_line - 1] += nl
    new_content = "".join(lines[:after_line]) + ins + "".join(lines[after_line:])
    _atomic_write(p, new_content)
    return f"Inserted after line {after_line} in {_rel(p)}\n{_diff(content, new_content, _rel(p))}"


@mcp.tool(annotations=WRITE)
def make_directory(path: str) -> str:
    """Create a directory (and any missing parents). Succeeds if it already exists."""
    _writable()
    p = _p(path)
    p.mkdir(parents=True, exist_ok=True)
    return f"Directory ready: {_rel(p)}"


@mcp.tool(annotations=DESTRUCTIVE)
def move_path(source: str, destination: str, overwrite: bool = False) -> str:
    """Move or rename a file or directory."""
    _writable()
    s, d = _p(source), _p(destination)
    if s == ROOT:
        raise ValueError("Cannot move the root directory.")
    if not s.exists():
        raise FileNotFoundError(f"Not found: {_rel(s)}")
    if d.exists() and d.is_dir() and not s.samefile(d):
        d = d / s.name
    if d.exists() and not overwrite:
        raise FileExistsError(f"{_rel(d)} already exists. Use overwrite=true.")
    if d.exists() and overwrite:
        shutil.rmtree(d) if d.is_dir() else d.unlink()
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(s), str(d))
    return f"Moved {_rel(s)} -> {_rel(d)}"


@mcp.tool(annotations=WRITE)
def copy_path(source: str, destination: str, overwrite: bool = False) -> str:
    """Copy a file or directory."""
    _writable()
    s, d = _p(source), _p(destination)
    if not s.exists():
        raise FileNotFoundError(f"Not found: {_rel(s)}")
    if d.exists() and d.is_dir() and s.is_file():
        d = d / s.name
    if d.exists() and not overwrite:
        raise FileExistsError(f"{_rel(d)} already exists. Use overwrite=true.")
    d.parent.mkdir(parents=True, exist_ok=True)
    if s.is_dir():
        shutil.copytree(s, d, dirs_exist_ok=overwrite)
    else:
        shutil.copy2(s, d)
    return f"Copied {_rel(s)} -> {_rel(d)}"


@mcp.tool(annotations=DESTRUCTIVE)
def delete_path(path: str, recursive: bool = False) -> str:
    """Delete a file or directory. Non-empty directories require recursive=true. The root cannot be deleted."""
    _writable()
    p = _p(path)
    if p == ROOT:
        raise ValueError("Refusing to delete the root directory.")
    if not p.exists() and not p.is_symlink():
        raise FileNotFoundError(f"Not found: {_rel(p)}")
    if p.is_dir() and not p.is_symlink():
        if any(p.iterdir()):
            if not recursive:
                raise OSError(f"{_rel(p)} is not empty. Use recursive=true to delete it and its contents.")
            shutil.rmtree(p)
        else:
            p.rmdir()
    else:
        p.unlink()
    return f"Deleted {_rel(p)}"


@mcp.tool(annotations=RO)
def get_server_info() -> str:
    """Show the server's root directory and mode."""
    return f"root: {ROOT}\nread_only: {READ_ONLY}\nmax_read_bytes: {MAX_READ_BYTES}\nmax_write_bytes: {MAX_WRITE_BYTES}"


# --------------------------------------------------------------------------- command execution
# NOTE: commands run as the user who started this server (e.g. your WSL Ubuntu user).
# The folder sandbox applies to the file tools and to the *starting directory* of commands,
# but a shell command can of course touch anything that user can. Disable with MCP_ENABLE_EXEC=0.
EXEC_ENABLED = os.environ.get("MCP_ENABLE_EXEC", "1").lower() in ("1", "true", "yes")
EXEC_TIMEOUT = int(os.environ.get("MCP_EXEC_TIMEOUT", "120"))
EXEC_MAX_TIMEOUT = int(os.environ.get("MCP_EXEC_MAX_TIMEOUT", "3600"))
MAX_OUTPUT = int(os.environ.get("MCP_MAX_OUTPUT", "60000"))
_STREAM_CAP = 5_000_000  # bytes kept in memory per stream
_HIDDEN_ENV = {"MCP_SECRET", "CLOUDFLARE_TUNNEL_TOKEN"}  # never leak these to child processes
EXEC = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True)
PROCS: dict[str, dict] = {}
_POSIX = os.name == "posix"

INTERPRETERS = {
    ".py": [sys.executable], ".sh": ["bash"], ".bash": ["bash"], ".js": ["node"], ".mjs": ["node"],
    ".rb": ["ruby"], ".pl": ["perl"], ".php": ["php"], ".ps1": ["pwsh", "-File"],
}


def _exec_allowed() -> None:
    if READ_ONLY:
        raise PermissionError("Server is in read-only mode (MCP_READ_ONLY=1); command execution is disabled.")
    if not EXEC_ENABLED:
        raise PermissionError("Command execution is disabled (MCP_ENABLE_EXEC=0).")


def _child_env(extra: dict | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in _HIDDEN_ENV}
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def _shell_argv(command: str) -> list[str]:
    sh = shutil.which("bash") or shutil.which("sh") or "/bin/sh"
    return [sh, "-c", command]


def _kill_tree(pid: int, sig=None) -> None:
    sig = sig or getattr(signal, "SIGKILL", signal.SIGTERM)
    try:
        if _POSIX:
            os.killpg(os.getpgid(pid), sig)
        else:
            os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _clip(s: str) -> str:
    if len(s) <= MAX_OUTPUT:
        return s
    half = MAX_OUTPUT // 2
    return s[:half] + f"\n...[{len(s) - MAX_OUTPUT} chars truncated]...\n" + s[-half:]


async def _run(argv: list[str], cwd: Path, timeout: int, env: dict) -> str:
    timeout = max(1, min(int(timeout), EXEC_MAX_TIMEOUT))
    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(cwd), env=env,
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=_POSIX)
    except FileNotFoundError:
        raise FileNotFoundError(f"Executable not found: {argv[0]}")

    buf = {"out": bytearray(), "err": bytearray()}

    async def pump(stream, key):
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                return
            if len(buf[key]) < _STREAM_CAP:
                buf[key] += chunk

    timed_out = False
    try:
        await asyncio.wait_for(
            asyncio.gather(pump(proc.stdout, "out"), pump(proc.stderr, "err"), proc.wait()), timeout)
    except asyncio.TimeoutError:
        timed_out = True
        _kill_tree(proc.pid)
        try:
            await asyncio.wait_for(proc.wait(), 5)
        except asyncio.TimeoutError:
            pass
    except asyncio.CancelledError:
        _kill_tree(proc.pid)
        raise

    out = buf["out"].decode("utf-8", "replace").rstrip()
    err = buf["err"].decode("utf-8", "replace").rstrip()
    parts = [
        f"exit_code: {proc.returncode}" + (f" (TIMED OUT after {timeout}s, process killed)" if timed_out else ""),
        f"duration: {time.monotonic() - started:.1f}s",
        f"cwd: {_rel(cwd)}",
    ]
    if out:
        parts.append("--- stdout ---\n" + _clip(out))
    if err:
        parts.append("--- stderr ---\n" + _clip(err))
    if not out and not err:
        parts.append("(no output)")
    return "\n".join(parts)


@mcp.tool(annotations=EXEC)
async def run_command(command: str, cwd: str = ".", timeout: int = EXEC_TIMEOUT,
                      env: dict[str, str] | None = None) -> str:
    """Run a shell command with bash on the machine hosting this server (Ubuntu/WSL) and return exit code,
    stdout and stderr. cwd is relative to the root folder. timeout is in seconds (the process is killed after it).
    Use start_process instead for servers or anything long-running. Commands run non-interactively (no stdin)."""
    _exec_allowed()
    d = _p(cwd)
    if not d.is_dir():
        raise NotADirectoryError(f"cwd is not a directory: {_rel(d)}")
    return await _run(_shell_argv(command), d, timeout, _child_env(env))


@mcp.tool(annotations=EXEC)
async def run_script(path: str, args: list[str] | None = None, interpreter: str = "auto", cwd: str = ".",
                     timeout: int = EXEC_TIMEOUT, env: dict[str, str] | None = None) -> str:
    """Run a script file (.py .sh .js .rb .pl .php .ps1, or any executable file) and return its output.
    interpreter='auto' picks one from the extension (.py uses the server's own Python; pass e.g.
    interpreter='python3' or '/path/to/venv/bin/python' to use another). cwd is relative to the root folder."""
    _exec_allowed()
    p = _p(path)
    if not p.is_file():
        raise FileNotFoundError(f"Script not found: {_rel(p)}")
    d = _p(cwd)
    if not d.is_dir():
        raise NotADirectoryError(f"cwd is not a directory: {_rel(d)}")
    if interpreter and interpreter != "auto":
        prefix = shlex.split(interpreter)
    else:
        prefix = INTERPRETERS.get(p.suffix.lower())
        if prefix is None:
            if not os.access(p, os.X_OK):
                raise ValueError(f"Unknown script type '{p.suffix}'. Pass interpreter=... or make the file executable.")
            prefix = []
    if prefix and not (os.path.isabs(prefix[0]) or shutil.which(prefix[0])):
        raise FileNotFoundError(f"Interpreter not installed: {prefix[0]}")
    return await _run([*prefix, str(p), *(args or [])], d, timeout, _child_env(env))


def _proc_state(rec: dict) -> str:
    rc = rec["popen"].poll()
    return "running" if rc is None else f"exited (code {rc})"


def _tail(path: Path, n: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    return _clip("\n".join(lines[-n:] if n > 0 else lines))


@mcp.tool(annotations=EXEC)
async def start_process(command: str, cwd: str = ".", env: dict[str, str] | None = None) -> str:
    """Start a long-running command in the background (dev server, watcher, training job...) and return a
    process_id. Check it with get_process_output / list_processes and stop it with stop_process."""
    _exec_allowed()
    d = _p(cwd)
    if not d.is_dir():
        raise NotADirectoryError(f"cwd is not a directory: {_rel(d)}")
    logdir = Path(tempfile.gettempdir()) / "mcp-procs"
    logdir.mkdir(exist_ok=True)
    pid = uuid.uuid4().hex[:8]
    log = logdir / f"{pid}.log"
    with open(log, "wb") as f:
        popen = subprocess.Popen(
            _shell_argv(command), cwd=str(d), env=_child_env(env), stdin=subprocess.DEVNULL,
            stdout=f, stderr=subprocess.STDOUT, start_new_session=_POSIX)
    PROCS[pid] = {"popen": popen, "command": command, "cwd": d, "log": log, "started": time.time()}
    await asyncio.sleep(1.0)  # let it fail fast / print startup output
    return (f"process_id: {pid}\nstatus: {_proc_state(PROCS[pid])}\nos_pid: {popen.pid}\n"
            f"--- output so far ---\n{_tail(log, 30) or '(none yet)'}")


@mcp.tool(annotations=RO)
def get_process_output(process_id: str, tail_lines: int = 100) -> str:
    """Get the status and the last N lines of combined output of a background process (tail_lines=0 for all)."""
    rec = PROCS.get(process_id)
    if not rec:
        raise KeyError(f"Unknown process_id: {process_id}. Use list_processes.")
    return f"status: {_proc_state(rec)}\ncommand: {rec['command']}\n--- output ---\n{_tail(rec['log'], tail_lines) or '(none)'}"


@mcp.tool(annotations=RO)
def list_processes() -> str:
    """List background processes started with start_process."""
    if not PROCS:
        return "No background processes."
    return "\n".join(
        f"{pid}  {_proc_state(r):<18} up {int(time.time() - r['started'])}s  cwd={_rel(r['cwd'])}  cmd={r['command'][:120]}"
        for pid, r in PROCS.items())


@mcp.tool(annotations=EXEC)
async def stop_process(process_id: str, force: bool = False) -> str:
    """Stop a background process (and its children). Sends SIGTERM, then SIGKILL after 8s; force=true kills at once."""
    _exec_allowed()
    rec = PROCS.get(process_id)
    if not rec:
        raise KeyError(f"Unknown process_id: {process_id}.")
    popen = rec["popen"]
    if popen.poll() is None:
        _kill_tree(popen.pid, None if force else signal.SIGTERM)
        for _ in range(40):
            if popen.poll() is not None:
                break
            await asyncio.sleep(0.2)
        if popen.poll() is None:
            _kill_tree(popen.pid)
            await asyncio.sleep(0.5)
    return f"{process_id}: {_proc_state(rec)}\n--- last output ---\n{_tail(rec['log'], 20) or '(none)'}"


@atexit.register
def _cleanup_processes() -> None:
    for rec in PROCS.values():
        if rec["popen"].poll() is None:
            _kill_tree(rec["popen"].pid)


@mcp.tool(annotations=RO)
def system_info() -> str:
    """Describe the host: OS, whether it is WSL, Python version and which common tools are installed."""
    rel = platform.release()
    wsl = "microsoft" in rel.lower()
    tools = ["python3", "pip", "node", "npm", "git", "docker", "bash", "curl", "powershell.exe"]
    found = {t: (shutil.which(t) or "not found") for t in tools}
    lines = [
        f"os: {platform.system()} {rel}",
        f"wsl: {wsl}",
        f"python: {sys.version.split()[0]} ({sys.executable})",
        f"user: {os.environ.get('USER') or os.environ.get('USERNAME') or '?'}",
        f"root: {ROOT}",
        f"exec_enabled: {EXEC_ENABLED and not READ_ONLY}",
        "tools:",
    ] + [f"  {k}: {v}" for k, v in found.items()]
    return "\n".join(lines)


# --------------------------------------------------------------------------- ASGI wrapper (secret path auth)
class SecretPathGate:
    """
    Only forwards requests to the MCP app when they arrive at  /<SECRET>/mcp
    (or at /mcp with 'Authorization: Bearer <SECRET>'). Everything else is 404.
    """

    def __init__(self, app, secret: str):
        if len(secret) < 16:
            raise SystemExit("MCP_SECRET must be set and at least 16 characters (run.py generates one).")
        self.app, self.secret = app, secret

    async def _respond(self, send, status: int, body: bytes = b"", ctype: bytes = b"text/plain"):
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", ctype), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)  # lifespan etc.

        path = scope.get("path", "")
        if path == "/healthz":
            return await self._respond(send, 200, b"ok")

        new_path = None
        parts = path.split("/", 2)  # ['', first, rest]
        if len(parts) >= 2 and parts[1] and hmac.compare_digest(parts[1].encode(), self.secret.encode()):
            new_path = "/mcp"
        else:
            auth = dict(scope.get("headers", [])).get(b"authorization", b"").decode()
            if auth.lower().startswith("bearer ") and hmac.compare_digest(auth[7:].strip().encode(), self.secret.encode()):
                new_path = "/mcp"

        if new_path is None:
            return await self._respond(send, 404, b"Not found")

        scope = dict(scope, path=new_path, raw_path=new_path.encode())
        return await self.app(scope, receive, send)


app = SecretPathGate(mcp.streamable_http_app(), SECRET) if SECRET else None

if __name__ == "__main__":
    raise SystemExit("Start with:  python run.py")