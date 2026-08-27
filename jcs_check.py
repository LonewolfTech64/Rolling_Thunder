from pathlib import Path

ROOT = Path("C:/Users/dalea")

for path in ROOT.rglob("*.jsonl"):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "CIA_" in text or "JCS" in text or "SNIE" in text:
            print("FOUND CORPUS:", path)
    except:
        pass
