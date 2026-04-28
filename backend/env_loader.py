import os
from pathlib import Path


ENV_LOADED = False


def find_env_path():
    backend_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / ".env",
        backend_dir.parent / ".env",
        backend_dir / ".env",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def parse_env_value(value):
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1]
    return cleaned


def load_env_file(path):
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, parse_env_value(value))


def load_env():
    global ENV_LOADED
    if ENV_LOADED:
        return

    env_path = find_env_path()
    try:
        from dotenv import load_dotenv
    except Exception:
        load_env_file(env_path)
    else:
        load_dotenv(env_path)
    ENV_LOADED = True
