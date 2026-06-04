import argparse
import os
import re
import pandas as pd

WORD = re.compile(r"\b\w+\b", re.UNICODE)

def simple_features(text: str):
    toks = WORD.findall(text.lower())
    n_tokens = len(toks)
    n_types = len(set(toks))
    ttr = n_types / max(n_tokens, 1)          
    avg_len = sum(len(w) for w in toks) / max(n_tokens, 1)
    rep_ratio = sum(
        1 for i in range(1, n_tokens) if toks[i] == toks[i - 1]
    ) / max(n_tokens, 1)

    return dict(
        n_tokens=n_tokens,
        n_types=n_types,
        ttr=ttr,
        avg_len=avg_len,
        rep_ratio=rep_ratio,
    )

def main(trans_dir, ids_file, labels_csv, out_csv):
    df_labels = pd.read_csv(labels_csv)  # columnas: id,label
    label_map = dict(zip(df_labels["id"], df_labels["label"]))

    rows = []
    with open(ids_file, encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]

    for sample_id in ids:
        txt_path = os.path.join(trans_dir, sample_id + ".txt")
        if not os.path.exists(txt_path):
            print(f"[!] No existe transcript para {sample_id}, lo salto.")
            continue

        text = open(txt_path, encoding="utf-8").read()
        feats = simple_features(text)
        feats["id"] = sample_id
        feats["label"] = label_map.get(sample_id, None)
        rows.append(feats)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[+] Guardado: {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trans_dir", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.trans_dir, args.ids, args.labels_csv, args.out)
