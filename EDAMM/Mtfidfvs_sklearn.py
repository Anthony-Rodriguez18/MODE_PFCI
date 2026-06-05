from pathlib import Path
import re
import json
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# RUTAS GENERALES
BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS = {
    "chat": BASE_DIR / "data" / "processed" / "dataset_chat_split.csv",
    "whisper": BASE_DIR / "data" / "processed" / "dataset_whisper_split.csv",
}

OUT_DIR = BASE_DIR / "outputs" / "tfidf_logreg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# STOPWORDS BÁSICAS
ENGLISH_STOPWORDS = {
    "a", "an", "the",
    "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "as", "by",
    "at", "from", "this", "that", "these", "those",
    "it", "its", "there", "here",
    "i", "you", "he", "she", "they", "we",
    "me", "him", "her", "them", "us",
    "my", "your", "his", "their", "our",
    "do", "does", "did",
    "have", "has", "had",
    "not", "no",
}


# IMPLEMENTACIÓN PROPIA DE TF-IDF
class MyTfidfVectorizer:
    def __init__(self, max_features=192, remove_stopwords=True, min_token_len=2):
        self.max_features = max_features
        self.remove_stopwords = remove_stopwords
        self.min_token_len = min_token_len

        self.vocabulary_ = {}
        self.idf_ = None
        self.feature_names_ = []

    def _tokenize(self, text):
        text = str(text).lower()

        text = re.sub(r"[^a-zA-Z'\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens = text.split()

        cleaned_tokens = []

        for token in tokens:
            if len(token) < self.min_token_len:
                continue

            if self.remove_stopwords and token in ENGLISH_STOPWORDS:
                continue

            cleaned_tokens.append(token)

        return cleaned_tokens

    def fit(self, documents):
        tokenized_docs = []
        global_counts = Counter()
        document_frequency = defaultdict(int)

        for doc in documents:
            tokens = self._tokenize(doc)
            tokenized_docs.append(tokens)

            global_counts.update(tokens)

            unique_tokens = set(tokens)
            for token in unique_tokens:
                document_frequency[token] += 1

        most_common = global_counts.most_common(self.max_features)

        self.feature_names_ = [word for word, _ in most_common]
        self.vocabulary_ = {
            word: idx for idx, word in enumerate(self.feature_names_)
        }

        n_documents = len(documents)

        self.idf_ = np.zeros(len(self.feature_names_), dtype=np.float64)

        for word, idx in self.vocabulary_.items():
            df_word = document_frequency[word]

            # Fórmula IDF:
            # IDF(w) = log((N + 1) / (DF(w) + 1)) + 1
            self.idf_[idx] = math.log((n_documents + 1) / (df_word + 1)) + 1

        return self

    def _transform_one_document(self, document):
        tokens = self._tokenize(document)

        vector = np.zeros(len(self.feature_names_), dtype=np.float64)

        if len(tokens) == 0:
            return vector

        counts = Counter(tokens)
        total_words = len(tokens)

        for word, count in counts.items():
            if word not in self.vocabulary_:
                continue

            idx = self.vocabulary_[word]

            # Fórmula TF:
            # TF(w,d) = count(w,d) / total_words(d)
            tf = count / total_words

            # Fórmula TF-IDF:
            # TF-IDF(w,d) = TF(w,d) * IDF(w)
            vector[idx] = tf * self.idf_[idx]

        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        return vector

    def transform(self, documents):
        matrix = []

        for doc in documents:
            vector = self._transform_one_document(doc)
            matrix.append(vector)

        return np.vstack(matrix)

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names_out(self):
        return np.array(self.feature_names_)

# IMPLEMENTACIÓN PROPIA DE REGRESIÓN LOGÍSTICA
class MyLogisticRegression:
    def __init__(self,learning_rate=0.5,epochs=1000,l2=0.0,verbose=True,print_every=100,
        random_state=42,):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.verbose = verbose
        self.print_every = print_every
        self.random_state = random_state

        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def _sigmoid(self, z):
      #Fórmula:
            #sigmoid(z) = 1 / (1 + exp(-z))

        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _forward(self, X):
        #Fórmula:
            #z = Xw + b
            #p = sigmoid(z)

        z = np.dot(X, self.weights) + self.bias
        p = self._sigmoid(z)
        return p

    def _compute_loss(self, y_true, y_pred):
        
        #Calcula Binary Cross Entropy.
        #Fórmula:
        #    L = -mean(y log(p) + (1-y) log(1-p))
        #Si l2 > 0, añade regularización:
        #    L_total = L + l2 * sum(w^2) / (2m)
        
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        m = len(y_true)

        bce = -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

        if self.l2 > 0:
            l2_penalty = (self.l2 / (2 * m)) * np.sum(self.weights ** 2)
            return bce + l2_penalty

        return bce

    def fit(self, X, y, X_valid=None, y_valid=None):
        np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        self.loss_history = []

        y = y.astype(np.float64)

        for epoch in range(1, self.epochs + 1):
            # 1. Forward
            y_pred = self._forward(X)

            # 2. Loss
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # 3. Error
            error = y_pred - y

            # 4. Gradientes
            # dw = X.T (p - y) / m
            dw = np.dot(X.T, error) / n_samples

            # db = mean(p - y)
            db = np.mean(error)

            # Regularización L2:
            # dw = dw + (l2/m) * w
            if self.l2 > 0:
                dw += (self.l2 / n_samples) * self.weights

            # 5. Actualización
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.verbose and epoch % self.print_every == 0:
                msg = f"Epoch {epoch}/{self.epochs} - train_loss={loss:.4f}"

                if X_valid is not None and y_valid is not None:
                    valid_pred = self._forward(X_valid)
                    valid_loss = self._compute_loss(y_valid, valid_pred)
                    msg += f" - valid_loss={valid_loss:.4f}"

                print(msg)

        return self

    def predict_proba(self, X):
        return self._forward(X)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


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

# CARGA DE SPLITS
def load_split_data(csv_path):
    df = pd.read_csv(csv_path)

    df["transcript"] = df["transcript"].fillna("").astype(str)
    df["label_id"] = df["label_id"].astype(int)

    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    test_df = df[df["split"] == "test"].copy()

    return train_df, valid_df, test_df


# EXPERIMENTO 1: TF-IDF PROPIO + LOGREG PROPIA
def run_mtfidf(dataset_name, csv_path):
    print("\n====================================================")
    print(f"EXPERIMENTO PROPIO: {dataset_name.upper()}")
    print("TF-IDF propio + Regresión Logística propia")
    print("====================================================")

    train_df, valid_df, test_df = load_split_data(csv_path)

    print("Train:", train_df.shape)
    print("Valid:", valid_df.shape)
    print("Test :", test_df.shape)

    # TF-IDF propio
    vectorizer = MyTfidfVectorizer(max_features=192,remove_stopwords=True)

    # fit_transform SOLO en train.
    X_train = vectorizer.fit_transform(train_df["transcript"])
    X_valid = vectorizer.transform(valid_df["transcript"])
    X_test = vectorizer.transform(test_df["transcript"])

    y_train = train_df["label_id"].values
    y_valid = valid_df["label_id"].values
    y_test = test_df["label_id"].values

    print("X_train:", X_train.shape)
    print("X_valid:", X_valid.shape)
    print("X_test :", X_test.shape)

    #Regresión Logística propia
    clf = MyLogisticRegression(learning_rate=0.5,epochs=1000, l2=0.1,verbose=True,
        print_every=200,)

    clf.fit(X_train,y_train,X_valid=X_valid,y_valid=y_valid)

    # Evaluación
    valid_pred = clf.predict(X_valid)
    test_pred = clf.predict(X_test)

    valid_metrics = print_metrics("VALID - implementación propia",y_valid,valid_pred)

    test_metrics = print_metrics("TEST - implementación propia", y_test,test_pred)

    #Guardar resultados
    result = {
        "dataset": dataset_name,
        "implementation": "own_tfidf_own_logreg",
        "max_features": 192,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }

    out_path = OUT_DIR / f"{dataset_name}_own_tfidf_own_logreg.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print("Guardado:", out_path)

    #Guardar vocabulario y pesos interpretables
    feature_names = vectorizer.get_feature_names_out()
    weights = clf.weights

    top_ad_idx = np.argsort(weights)[-20:][::-1]
    top_nonad_idx = np.argsort(weights)[:20]

    top_words = {
        "top_ad_words": [
            {"word": str(feature_names[i]), "weight": float(weights[i])}
            for i in top_ad_idx
        ],
        "top_nonad_words": [
            {"word": str(feature_names[i]), "weight": float(weights[i])}
            for i in top_nonad_idx
        ],
    }

    out_words = OUT_DIR / f"{dataset_name}_own_tfidf_own_logreg_top_words.json"

    with open(out_words, "w", encoding="utf-8") as f:
        json.dump(top_words, f, indent=4)

    print("Pesos interpretables guardados:", out_words)

# EXPERIMENTO 2: SKLEARN TF-IDF + SKLEARN LOGREG
def run_sklearn(dataset_name, csv_path):
    print("\n====================================================")
    print(f"EXPERIMENTO SKLEARN: {dataset_name.upper()}")
    print("TF-IDF sklearn + Regresión Logística sklearn")
    print("====================================================")

    train_df, valid_df, test_df = load_split_data(csv_path)

    print("Train:", train_df.shape)
    print("Valid:", valid_df.shape)
    print("Test :", test_df.shape)

    vectorizer = TfidfVectorizer(max_features=192,lowercase=True,stop_words="english",
        ngram_range=(1, 1))

    # fit_transform SOLO en train.
    X_train = vectorizer.fit_transform(train_df["transcript"])
    X_valid = vectorizer.transform(valid_df["transcript"])
    X_test = vectorizer.transform(test_df["transcript"])

    y_train = train_df["label_id"].values
    y_valid = valid_df["label_id"].values
    y_test = test_df["label_id"].values

    clf = LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42)

    clf.fit(X_train, y_train)

    valid_pred = clf.predict(X_valid)
    test_pred = clf.predict(X_test)

    valid_metrics = print_metrics("VALID - sklearn",y_valid,valid_pred)

    test_metrics = print_metrics("TEST - sklearn",y_test,test_pred)

    result = {
        "dataset": dataset_name,
        "implementation": "sklearn_tfidf_sklearn_logreg",
        "max_features": 192,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }

    out_path = OUT_DIR / f"{dataset_name}_sklearn_tfidf_sklearn_logreg.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print("Guardado:", out_path)

    feature_names = vectorizer.get_feature_names_out()
    weights = clf.coef_[0]

    top_ad_idx = np.argsort(weights)[-20:][::-1]
    top_nonad_idx = np.argsort(weights)[:20]

    top_words = {
        "top_ad_words": [
            {"word": str(feature_names[i]), "weight": float(weights[i])}
            for i in top_ad_idx
        ],
        "top_nonad_words": [
            {"word": str(feature_names[i]), "weight": float(weights[i])}
            for i in top_nonad_idx
        ],
    }

    out_words = OUT_DIR / f"{dataset_name}_sklearn_tfidf_sklearn_logreg_top_words.json"

    with open(out_words, "w", encoding="utf-8") as f:
        json.dump(top_words, f, indent=4)

    print("Pesos interpretables guardados:", out_words)

def main():
    for dataset_name, csv_path in DATASETS.items():
        if not csv_path.exists():
            print("No existe:", csv_path)
            continue

        run_mtfidf(dataset_name, csv_path)
        run_sklearn(dataset_name, csv_path)


if __name__ == "__main__":
    main()