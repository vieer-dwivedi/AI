import subprocess

def run_command(cmd, cwd):
    return subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=True, text=True, timeout=60
    )
