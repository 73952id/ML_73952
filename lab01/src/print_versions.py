import importlib.metadata as metadata
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
print('Python:', platform.python_version())
for line in (ROOT / 'requirements.txt').read_text().splitlines():
    if '==' not in line:
        continue
    name, expected = line.split('==', 1)
    try:
        actual = metadata.version(name)
    except metadata.PackageNotFoundError:
        actual = 'NOT INSTALLED'
    print(f'{name}: {actual} (required: {expected})')
