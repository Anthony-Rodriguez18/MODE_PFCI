import argparse
import os
import whisper

def main(audio_dir, out_dir, model_name="large-v2"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[+] Cargando modelo Whisper: {model_name}")
    model = whisper.load_model(model_name)

    for fn in sorted(os.listdir(audio_dir)):
        if not fn.lower().endswith((".wav", ".flac", ".mp3", ".m4a")):
            continue

        audio_path = os.path.join(audio_dir, fn)
        print(f"[+] Transcribiendo: {audio_path}")
        result = model.transcribe(audio_path, verbose=False)
        text = result["text"].strip()

        base = os.path.splitext(fn)[0]
        out_path = os.path.join(out_dir, base + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="large-v2")
    args = ap.parse_args()
    main(args.audio_dir, args.out_dir, args.model)
