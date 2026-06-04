import argparse
import os
import json
import time
import pandas as pd
from openai import OpenAI

client = OpenAI()  # lee OPENAI_API_KEY

SYSTEM_MSG = """
Eres un clínico experto en lenguaje y demencias.
Vas a puntuar un discurso espontáneo en 5 indicadores de deterioro cognitivo:
1) word_finding_difficulties (WFD)
2) semantic_paraphasias
3) syntactic_simplification
4) impoverished_vocabulary
5) discourse_impairment

Devuelve SOLO un JSON con esta estructura:

{
  "word_finding_difficulties":  {"score": <entero 1-7>},
  "semantic_paraphasias":      {"score": <entero 1-7>},
  "syntactic_simplification":  {"score": <entero 1-7>},
  "impoverished_vocabulary":   {"score": <entero 1-7>},
  "discourse_impairment":      {"score": <entero 1-7>}
}

No añadas texto fuera del JSON.
"""

USER_TEMPLATE = """
Transcripción de habla espontánea:

\"\"\"{transcript}\"\"\"

Tarea:
- Lee el texto.
- Puntúa cada indicador con un número ENTERO entre 1 (nada presente) y 7 (muy marcado).
- No expliques nada, solo devuelve el JSON pedido.
"""

def score_with_gpt(text: str):
    prompt = USER_TEMPLATE.format(transcript=text)

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",  # cámbialo a gpt-4o o gpt-4o-mini si tienes acceso
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
    )

    content = resp.choices[0].message.content
    data = json.loads(content)

    return dict(
        gpt_wfd=int(data["word_finding_difficulties"]["score"]),
        gpt_sempar=int(data["semantic_paraphasias"]["score"]),
        gpt_synsimp=int(data["syntactic_simplification"]["score"]),
        gpt_vocab=int(data["impoverished_vocabulary"]["score"]),
        gpt_disc=int(data["discourse_impairment"]["score"]),
    )

def main(ids_file, trans_dir, out_csv, sleep_sec=0.5):
    rows = []
    with open(ids_file, encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]

    for sample_id in ids:
        path = os.path.join(trans_dir, sample_id + ".txt")
        if not os.path.exists(path):
            print(f"[!] No existe transcript para {sample_id}, lo salto.")
            continue

        print(f"[+] GPT puntuando: {sample_id}")
        text = open(path, encoding="utf-8").read()

        try:
            scores = score_with_gpt(text)
        except Exception as e:
            print(f"[!] Error con {sample_id}: {e}")
            continue

        row = {"id": sample_id}
        row.update(scores)
        rows.append(row)

        time.sleep(sleep_sec)  # para no spamear la API

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[+] Guardado GPT-features: {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)         # data/splits/train.txt
    ap.add_argument("--trans_dir", required=True)   # data/transcripts
    ap.add_argument("--out", required=True)         # features/gpt/train.csv
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()
    main(args.ids, args.trans_dir, args.out, args.sleep)
