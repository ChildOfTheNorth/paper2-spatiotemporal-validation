from pathlib import Path
import hashlib, json
root = Path(__file__).resolve().parents[1]
for item in json.loads((root/'SHA256SUMS.json').read_text()):
    path = root/item['file']
    if path.stat().st_size != item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        raise SystemExit(f"FAILED: {item['file']}")
print('All snapshot hashes and sizes verified.')
