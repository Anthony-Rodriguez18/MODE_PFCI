import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "raw" / "pitt_cookie_audio"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "pitt_cookie_wav"

for group in ["Control", "Dementia"]:
    input_group_dir = INPUT_DIR / group
    output_group_dir = OUTPUT_DIR / group
    output_group_dir.mkdir(parents=True, exist_ok=True)

    mp3_files = list(input_group_dir.glob("*.mp3"))

    print(f"{group}: {len(mp3_files)} archivos MP3")

    for mp3_path in mp3_files:
        wav_path = output_group_dir / f"{mp3_path.stem}.wav"

        if wav_path.exists() and wav_path.stat().st_size > 0:
            print(f"[SKIP] {wav_path}")
            continue

        command = [
            "ffmpeg",
            "-y",
            "-i", str(mp3_path),
            "-ac", "1",
            "-ar", "16000",
            str(wav_path),
        ]

        print(f"[CONVERT] {mp3_path.name} -> {wav_path.name}")
        subprocess.run(command, check=True)

print("Conversión terminada.")