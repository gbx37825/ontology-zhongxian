from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
manifest = json.loads((root/'文件清单.json').read_text())
failed = []
for name, expected in manifest['files'].items():
    p = root/name
    if not p.is_file() or p.stat().st_size != expected['bytes']:
        failed.append(name)
        continue
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    if h.hexdigest() != expected['sha256']:
        failed.append(name)
print(json.dumps({'status': 'FAILED' if failed else 'PASSED', 'checked': len(manifest['files']), 'failed': failed}, ensure_ascii=False))
if failed:
    raise SystemExit(1)
