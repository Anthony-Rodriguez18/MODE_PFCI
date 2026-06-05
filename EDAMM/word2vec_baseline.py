from pathlib import Path
import re
import json

import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# RUTAS
BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS = {
    "chat": BASE_DIR / "data" / "processed" / "dataset_chat_split.csv",
    "whisper": BASE_DIR / "data" / "processed" / "dataset_whisper_split.csv",
}

OUT_DIR = BASE_DIR / "outputs" / "word2vec"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_DIR = BASE_DIR / "data" / "features" / "word2vec"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# CONFIGURACIÓN WORD2VEC
VECTOR_SIZE = 192
WINDOW = 5
MIN_COUNT = 1
WORKERS = 4
SG = 0  # 0 = CBOW, 1 = Skip-gram
EPOCHS = 100
RANDOM_STATE = 42


# ============================================================
# TOKENIZACIÓN
# ============================================================

def tokenize_text(text):
    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    clean_tokens = []

    for token in tokens:
        if len(token) < 2:
            continue

        if token in ENGLISH_STOP_WORDS:
            continue

        clean_tokens.append(token)

    return clean_tokens


# ============================================================
# DOCUMENT EMBEDDING
def document_vector(tokens, model, vector_size):
    #Convierte una lista de tokens en un vector de documento.

    vectors = []

    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])

    if len(vectors) == 0:
        return np.zeros(vector_size, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


def build_document_matrix(tokenized_docs, model, vector_size):
    #Convierte varios documentos tokenizados en una matriz.
    matrix = []

    for tokens in tokenized_docs:
        vec = document_vector(tokens, model, vector_size)
        matrix.append(vec)

    return np.vstack(matrix)


# ============================================================
# MÉTRICAS
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_metrics(title, y_true, y_pred):
    print("\n" + title)
    print("-" * len(title))

    metrics = compute_metrics(y_true, y_pred)

    print("Accuracy :", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall   :", metrics["recall"])
    print("F1       :", metrics["f1"])
    print("Confusion matrix:")
    print(np.array(metrics["confusion_matrix"]))

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Non-AD", "AD"],
        zero_division=0
    ))

    return metrics


# ============================================================
# CARGA DE SPLITS
def load_split_data(csv_path):
    df = pd.read_csv(csv_path)

    df["transcript"] = df["transcript"].fillna("").astype(str)
    df["label_id"] = df["label_id"].astype(int)

    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    test_df = df[df["split"] == "test"].copy()

    return train_df, valid_df, test_df


# ============================================================
# EXPERIMENTO WORD2VEC + LOGISTIC REGRESSION
def run_word2vec_experiment(dataset_name, csv_path):
    print("\n====================================================")
    print(f"EXPERIMENTO WORD2VEC: {dataset_name.upper()}")
    print("Word2Vec CBOW + Regresión Logística sklearn")
    print("====================================================")

    train_df, valid_df, test_df = load_split_data(csv_path)

    print("Train:", train_df.shape)
    print("Valid:", valid_df.shape)
    print("Test :", test_df.shape)

    #Tokenizar
    train_tokens = [tokenize_text(text) for text in train_df["transcript"]]
    valid_tokens = [tokenize_text(text) for text in valid_df["transcript"]]
    test_tokens = [tokenize_text(text) for text in test_df["transcript"]]

    #Entrenar Word2Vec SOLO con train
    model = Word2Vec(
        sentences=train_tokens,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=SG,
        epochs=EPOCHS,
        seed=RANDOM_STATE,
    )

    print("Tamaño vocabulario Word2Vec:", len(model.wv))
    print("Dimensión embedding:", VECTOR_SIZE)

    #Convertir documentos a vectores
    X_train = build_document_matrix(train_tokens, model, VECTOR_SIZE)
    X_valid = build_document_matrix(valid_tokens, model, VECTOR_SIZE)
    X_test = build_document_matrix(test_tokens, model, VECTOR_SIZE)

    y_train = train_df["label_id"].values
    y_valid = valid_df["label_id"].values
    y_test = test_df["label_id"].values

    print("X_train:", X_train.shape)
    print("X_valid:", X_valid.shape)
    print("X_test :", X_test.shape)

    #Guardar features para usarlas luego en EDAMM
    dataset_features_dir = FEATURES_DIR / dataset_name
    dataset_features_dir.mkdir(parents=True, exist_ok=True)

    np.save(dataset_features_dir / "X_train_word2vec.npy", X_train)
    np.save(dataset_features_dir / "X_valid_word2vec.npy", X_valid)
    np.save(dataset_features_dir / "X_test_word2vec.npy", X_test)

    np.save(dataset_features_dir / "y_train.npy", y_train)
    np.save(dataset_features_dir / "y_valid.npy", y_valid)
    np.save(dataset_features_dir / "y_test.npy", y_test)

    model.save(str(dataset_features_dir / "word2vec.model"))

    print("Features guardadas en:", dataset_features_dir)

    #Clasificador
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    clf.fit(X_train, y_train)

    valid_pred = clf.predict(X_valid)
    test_pred = clf.predict(X_test)

    valid_metrics = print_metrics(
        "VALID - Word2Vec + Logistic Regression",
        y_valid,
        valid_pred
    )

    test_metrics = print_metrics(
        "TEST - Word2Vec + Logistic Regression",
        y_test,
        test_pred
    )

    #Guardar resultados
    result = {
        "dataset": dataset_name,
        "model": "Word2Vec CBOW + Logistic Regression",
        "vector_size": VECTOR_SIZE,
        "window": WINDOW,
        "min_count": MIN_COUNT,
        "sg": SG,
        "epochs": EPOCHS,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }

    out_path = OUT_DIR / f"{dataset_name}_word2vec_logreg_metrics.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print("Resultados guardados:", out_path)


def main():
    for dataset_name, csv_path in DATASETS.items():
        if not csv_path.exists():
            print("No existe:", csv_path)
            continue

        run_word2vec_experiment(dataset_name, csv_path)


if __name__ == "__main__":
    main()