"""Per-video result rows of the evaluators (`eval.py`, `eval_lvos.py`): written atomically (a temporary file renamed into
place, so an interrupted worker never leaves a partial row), and checked on resume (an existing row must parse and
hold every requested setting; a row from other settings in the same output directory is an error, not a silent skip)."""
import json
import os
from pathlib import Path


def done(f: Path, wanted):
    """True if `f` holds a complete row for every setting in `wanted`; False if absent; raises if it is unreadable or
    was written for other settings (use a fresh output directory per setting list)."""
    if not f.exists():
        return False
    try:
        have = {r["setting"] for r in json.loads(f.read_text())}
    except (ValueError, KeyError, TypeError) as e:
        raise SystemExit(f"{f}: unreadable result row ({e}); remove it to recompute")
    missing = set(wanted) - have
    if missing:
        raise SystemExit(f"{f}: written for other settings (missing {sorted(missing)[:3]}); use a fresh output directory")
    return True


def write(f: Path, rows):
    tmp = f.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(rows))
    os.replace(tmp, f)
