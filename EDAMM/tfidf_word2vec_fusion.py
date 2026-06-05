from pathlib import Path
import re
import json

import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS = {
    "chat": BASE_DIR / "data" / "processed" / "dataset_chat_split.csv",
    "whisper": BASE_DIR / "data" / "processed" / "dataset_whisper_split.csv",
}

OUT_DIR = BASE_DIR / "outputs" / "tfidf_word2vec"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_DIR = BASE_DIR / "data" / "features" / "tfidf_word2vec"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURACIÓN
# ============================================================

TFIDF_MAX_FEATURES = 192

WORD2VEC_VECTOR_SIZE = 192
WORD2VEC_WINDOW = 5
WORD2VEC_MIN_COUNT = 1
WORD2VEC_WORKERS = 4
WORD2VEC_SG = 0  # 0 = CBOW, 1 = Skip-gram
WORD2VEC_EPOCHS = 100

RANDOM_STATE = 42


# ============================================================
# TOKENIZACIÓN PARA WORD2VEC
# ============================================================

def tokenize_text(text):
    """
    Tokeniza y limpia una transcripción para entrenar/usar Word2Vec.
    """
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
# WORD2VEC: DOCUMENT VECTOR
# ============================================================

def document_vector(tokens, model, vector_size):
    """
    Promedia embeddings Word2Vec para obtener un vector por documento.
    """
    vectors = []

    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])

    if len(vectors) == 0:
        return np.zeros(vector_size, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


def build_word2vec_matrix(tokenized_docs, model, vector_size):
    """
    Convierte documentos tokenizados a matriz [n_docs, vector_size].
    """
    matrix = []

    for tokens in tokenized_docs:
        vec = document_vector(tokens, model, vector_size)
        matrix.append(vec)

    return np.vstack(matrix)


# ============================================================
# MÉTRICAS
# ============================================================

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
# ============================================================

def load_split_data(csv_path):
    """
    Carga train/valid/test desde un CSV con columna split.
    """
    df = pd.read_csv(csv_path)

    df["transcript"] = df["transcript"].fillna("").astype(str)
    df["label_id"] = df["label_id"].astype(int)

    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    test_df = df[df["split"] == "test"].copy()

    return train_df, valid_df, test_df


# ============================================================
# EXPERIMENTO: TF-IDF + WORD2VEC

def run_experiment(dataset_name, csv_path):
    print("\n====================================================")
    print(f"EXPERIMENTO FUSIÓN TEXTUAL: {dataset_name.upper()}")
    print("TF-IDF + Word2Vec + Regresión Logística")
    print("====================================================")

    train_df, valid_df, test_df = load_split_data(csv_path)

    print("Train:", train_df.shape)
    print("Valid:", valid_df.shape)
    print("Test :", test_df.shape)

    #TF-IDF
    # (fit en train; transform en valid/test)

    tfidf_vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 1)
    )

    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df["transcript"]).toarray()
    X_valid_tfidf = tfidf_vectorizer.transform(valid_df["transcript"]).toarray()
    X_test_tfidf = tfidf_vectorizer.transform(test_df["transcript"]).toarray()

    print("TF-IDF train:", X_train_tfidf.shape)
    print("TF-IDF valid:", X_valid_tfidf.shape)
    print("TF-IDF test :", X_test_tfidf.shape)

    # WORD2VEC
    # (entrena en train; representa docs como promedio de embeddings)

    train_tokens = [tokenize_text(text) for text in train_df["transcript"]]
    valid_tokens = [tokenize_text(text) for text in valid_df["transcript"]]
    test_tokens = [tokenize_text(text) for text in test_df["transcript"]]

    word2vec_model = Word2Vec(
        sentences=train_tokens,
        vector_size=WORD2VEC_VECTOR_SIZE,
        window=WORD2VEC_WINDOW,
        min_count=WORD2VEC_MIN_COUNT,
        workers=WORD2VEC_WORKERS,
        sg=WORD2VEC_SG,
        epochs=WORD2VEC_EPOCHS,
        seed=RANDOM_STATE,
    )

    X_train_w2v = build_word2vec_matrix(
        train_tokens,
        word2vec_model,
        WORD2VEC_VECTOR_SIZE
    )

    X_valid_w2v = build_word2vec_matrix(
        valid_tokens,
        word2vec_model,
        WORD2VEC_VECTOR_SIZE
    )

    X_test_w2v = build_word2vec_matrix(
        test_tokens,
        word2vec_model,
        WORD2VEC_VECTOR_SIZE
    )

    print("Word2Vec vocab size:", len(word2vec_model.wv))
    print("Word2Vec train:", X_train_w2v.shape)
    print("Word2Vec valid:", X_valid_w2v.shape)
    print("Word2Vec test :", X_test_w2v.shape)
    # FUSIÓN TEXTUAL
    # Concatena TF-IDF y Word2Vec (features de texto fusionadas)

    X_train_fusion = np.concatenate(
        [X_train_tfidf, X_train_w2v],
        axis=1
    )

    X_valid_fusion = np.concatenate(
        [X_valid_tfidf, X_valid_w2v],
        axis=1
    )

    X_test_fusion = np.concatenate(
        [X_test_tfidf, X_test_w2v],
        axis=1
    )

    print("Fusion train:", X_train_fusion.shape)
    print("Fusion valid:", X_valid_fusion.shape)
    print("Fusion test :", X_test_fusion.shape)

    y_train = train_df["label_id"].values
    y_valid = valid_df["label_id"].values
    y_test = test_df["label_id"].values

    # GUARDAR FEATURES
    dataset_features_dir = FEATURES_DIR / dataset_name
    dataset_features_dir.mkdir(parents=True, exist_ok=True)

    np.save(dataset_features_dir / "X_train_tfidf.npy", X_train_tfidf)
    np.save(dataset_features_dir / "X_valid_tfidf.npy", X_valid_tfidf)
    np.save(dataset_features_dir / "X_test_tfidf.npy", X_test_tfidf)

    np.save(dataset_features_dir / "X_train_word2vec.npy", X_train_w2v)
    np.save(dataset_features_dir / "X_valid_word2vec.npy", X_valid_w2v)
    np.save(dataset_features_dir / "X_test_word2vec.npy", X_test_w2v)

    np.save(dataset_features_dir / "X_train_text_fusion.npy", X_train_fusion)
    np.save(dataset_features_dir / "X_valid_text_fusion.npy", X_valid_fusion)
    np.save(dataset_features_dir / "X_test_text_fusion.npy", X_test_fusion)

    np.save(dataset_features_dir / "y_train.npy", y_train)
    np.save(dataset_features_dir / "y_valid.npy", y_valid)
    np.save(dataset_features_dir / "y_test.npy", y_test)

    word2vec_model.save(str(dataset_features_dir / "word2vec.model"))

    print("Features guardadas en:", dataset_features_dir)

    # CLASIFICADOR
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    clf.fit(X_train_fusion, y_train)

    valid_pred = clf.predict(X_valid_fusion)
    test_pred = clf.predict(X_test_fusion)

    valid_metrics = print_metrics(
        "VALID - TF-IDF + Word2Vec",
        y_valid,
        valid_pred
    )

    test_metrics = print_metrics(
        "TEST - TF-IDF + Word2Vec",
        y_test,
        test_pred
    )

    # ========================================================
    # GUARDAR RESULTADOS
    # ========================================================

    result = {
        "dataset": dataset_name,
        "model": "TF-IDF + Word2Vec + Logistic Regression",
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "word2vec_vector_size": WORD2VEC_VECTOR_SIZE,
        "word2vec_window": WORD2VEC_WINDOW,
        "word2vec_min_count": WORD2VEC_MIN_COUNT,
        "word2vec_sg": WORD2VEC_SG,
        "word2vec_epochs": WORD2VEC_EPOCHS,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }

    out_path = OUT_DIR / f"{dataset_name}_tfidf_word2vec_logreg_metrics.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print("Resultados guardados:", out_path)


def main():
    for dataset_name, csv_path in DATASETS.items():
        if not csv_path.exists():
            print("No existe:", csv_path)
            continue

        run_experiment(dataset_name, csv_path)


if __name__ == "__main__":
    main()