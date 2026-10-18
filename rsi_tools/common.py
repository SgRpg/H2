from __future__ import annotations
import datetime as dt
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
VENDOR = ROOT / 'vendor' / 'evotool'
COMMIT = '6a82719ea6e22cf23f963078e2527094e50b47f6'
MODEL = 'Qwen/Qwen3-4B'
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
sys.dont_write_bytecode = True

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def read(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8-sig'))

def write_new(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def append(path, obj):
    with pathlib.Path(path).open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

def activate_vendor():
    verify_vendor()
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))

def verify_vendor():
    frozen = read(ROOT / 'baseline-manifest.json')
    bad = [name for name, expected in frozen['files'].items()
           if not (VENDOR / name).is_file() or sha(VENDOR / name) != expected]
    if bad:
        raise RuntimeError('Author baseline changed: ' + ', '.join(bad[:5]))
    return frozen

def environment():
    packages = {}
    for name in ['torch', 'transformers', 'accelerate', 'huggingface-hub', 'openai', 'PyYAML', 'lm-format-enforcer']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    inventory = sorted({(d.metadata.get('Name', 'unknown'), d.version) for d in importlib.metadata.distributions()})
    return {'python': sys.version, 'platform': platform.platform(), 'packages': packages,
            'installed_distribution_versions': dict(inventory)}

def new_run_dir(path):
    path = pathlib.Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path
