from pathlib import Path

base = Path("data/raw/pitt_cookie_audio")

for group in ["Control", "Dementia"]:
    files = list((base / group).glob("*.mp3"))
    print(group, len(files))