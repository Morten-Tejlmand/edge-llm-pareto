"""Read append-only trial files, recovering only an interrupted final write."""
import json
import warnings
from pathlib import Path


def read_records(path, repair=False):
    path = Path(path)
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows = []
    offset = 0
    for i, line in enumerate(lines):
        if not line.strip():
            offset += len(line)
            continue
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i != len(lines) - 1 or line.endswith(b'\n'):
                raise
            warnings.warn(f'{path}: ignoring incomplete final record', stacklevel=2)
            if repair:
                with path.open('r+b') as f:
                    f.truncate(offset)
            break
        offset += len(line)
    if repair and rows and raw and not raw.endswith(b'\n') and offset == len(raw):
        with path.open('ab') as f:
            f.write(b'\n')
    return rows
