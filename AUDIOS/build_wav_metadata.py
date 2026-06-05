import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

WAV_DIR = BASE_DIR / "data" / "processed" / "pitt_cookie_wav"
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "pitt_cookie_wav_metadata.csv"

rows = []

for group in ["Control", "Dementia"]:
    label_id = 0 if group == "Control" else 1
    label_name = "Non-AD" if group == "Control" else "AD"

    for wav_path in sorted((WAV_DIR / group).glob("*.wav")):
        rows.append({
            "sample_id": wav_path.stem,
            "group": group,
            "label_name": label_name,
            "label_id": label_id,
            "wav_path": str(wav_path).replace("\\", "/"),
        })

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["sample_id", "group", "label_name", "label_id", "wav_path"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Metadata guardada en: {OUTPUT_CSV}")
print(f"Total muestras: {len(rows)}")
print(f"Control: {sum(1 for r in rows if r['group'] == 'Control')}")
print(f"Dementia: {sum(1 for r in rows if r['group'] == 'Dementia')}")