"""Install the pinned official MTG model assets; no uploads or credentials."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "data/audio-models.manifest.json").read_text())
    target = ROOT / "models/musicnn"
    target.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        path = target / item["filename"]
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]:
            print("Verified:", path.name)
            continue
        temporary = path.with_suffix(path.suffix + ".download")
        try:
            with urllib.request.urlopen(item["url"], timeout=60) as response:
                data = response.read(item["bytes"] + 1)
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"Checksum/size mismatch: {path.name}")
            temporary.write_bytes(data)
            temporary.replace(path)
            print("Installed:", path.name)
        finally:
            temporary.unlink(missing_ok=True)
    print("Ready. Models: CC BY-NC-SA 4.0; see docs/audio-models.md for attribution.")


if __name__ == "__main__":
    main()
