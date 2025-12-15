import os

SANDBOX_DIR = os.path.abspath("agent_workspace")


def _safe_path(path: str) -> str:
    full_path = os.path.abspath(os.path.join(SANDBOX_DIR, path))
    if not full_path.startswith(SANDBOX_DIR):
        raise PermissionError("Access outside sandbox is not allowed.")
    return full_path


def write_file(path: str, content: str) -> str:
    full_path = _safe_path(path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"File written successfully: {path}"


def read_file(path: str) -> str:
    full_path = _safe_path(path)
    if not os.path.exists(full_path):
        return "File does not exist."

    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


def list_files(path: str = ".") -> str:
    full_path = _safe_path(path)
    if not os.path.exists(full_path):
        return "Directory does not exist."

    return "\n".join(os.listdir(full_path))
