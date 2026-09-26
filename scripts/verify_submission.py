"""Check the distribution's integrity and common credential-leak patterns.

Diagnostics contain paths and rule names, NEVER matching values or source lines.
This heuristic scan cannot prove the absence of every possible secret.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "provider_token": re.compile(rb"(?<![A-Za-z0-9])(?:sk-[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{30,}|gh[pousr]_[A-Za-z0-9]{25,})"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "bearer_literal": re.compile(rb"Bearer\s+[A-Za-z0-9_.-]{24,}"),
    "credential_url": re.compile(rb"https?://[^\s/@:]+:[^\s/@]+@"),
    "author_home_path": re.compile(rb"/home/[A-Za-z0-9_.-]+/"),
}
SAFE_LITERALS = {"", "offline-test-key", "test-key", "dummy", "none", "your-api-key", "your_api_key"}
SENSITIVE_NAME = re.compile(r"(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)", re.I)


def files(root=ROOT):
    return sorted(p for p in root.rglob('*') if p.is_file() and not {'.git', '__pycache__', '.pytest_cache'}.intersection(p.parts))


def scan(root=ROOT):
    issues=[]
    for p in files(root):
        name=p.relative_to(root).as_posix(); data=p.read_bytes()
        if p.name == 'python_settings_local.py' or (p.name.startswith('.env') and p.name != '.env.example'):
            issues.append({'path':name,'rule':'private_config_file'})
        if p.suffix in {'.csv','.tsv','.jsonl','.npy','.npz','.h5ad','.pkl','.pt','.pth','.ckpt','.log','.xlsx'}:
            issues.append({'path':name,'rule':'excluded_data_or_binary'})
        for rule,pattern in PATTERNS.items():
            if pattern.search(data): issues.append({'path':name,'rule':rule})
        if p.suffix != '.py': continue
        try: tree=ast.parse(data.decode('utf-8'))
        except (SyntaxError,UnicodeError):
            issues.append({'path':name,'rule':'invalid_python'}); continue
        pairs=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign):
                pairs.extend((ast.unparse(t),node.value) for t in node.targets)
            elif isinstance(node,ast.Dict):
                pairs.extend((k.value,v) for k,v in zip(node.keys,node.values) if isinstance(k,ast.Constant) and isinstance(k.value,str))
            elif isinstance(node,ast.keyword) and node.arg:
                pairs.append((node.arg,node.value))
        for key,value in pairs:
            if SENSITIVE_NAME.search(key) and isinstance(value,ast.Constant) and isinstance(value.value,str):
                if value.value.lower() not in SAFE_LITERALS and len(value.value)>5:
                    issues.append({'path':name,'rule':'credential_literal','line':value.lineno})
    return issues


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--skip-integrity',action='store_true',help='Use only while preparing a new release.')
    args=ap.parse_args()
    issues=scan()
    manifest=ROOT/'checksums.json'
    if not args.skip_integrity:
        if not manifest.exists(): issues.append({'rule':'missing_checksums'})
        else:
            expected=json.loads(manifest.read_text())
            actual={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files() if p!=manifest}
            for name in sorted(set(actual)|set(expected)):
                if actual.get(name)!=expected.get(name): issues.append({'path':name,'rule':'integrity_mismatch'})
    print(json.dumps({'status':'PASS' if not issues else 'FAIL','files_checked':len(files()),'findings':issues},indent=2))
    raise SystemExit(bool(issues))


if __name__=='__main__':main()
