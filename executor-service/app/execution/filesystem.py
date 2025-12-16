import os

def ensure_workdir(base, workflow_id):
    path = os.path.join(base, workflow_id)
    os.makedirs(path, exist_ok=True)
    return path

def write_file(workdir, rel_path, content):
    full = os.path.join(workdir, rel_path)
    os.makedirs(os.path.dirname(full) or workdir, exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return full
