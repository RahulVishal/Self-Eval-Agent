"""
Baseline model for the depression-detection project: TF-IDF + Logistic Regression.

Run after data_pipeline.py has produced train.csv / val.csv / test.csv.

    python baseline_model.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import joblib

TEXT_COL = "clean_text"
LABEL_COL = "is_depression"


def load_splits():
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")
    test_df = pd.read_csv("test.csv")
    return train_df, val_df, test_df


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
        sublinear_tf=True,
    )


def evaluate(model, vectorizer, df: pd.DataFrame, split_name: str):
    X = vectorizer.transform(df[TEXT_COL])
    y_true = df[LABEL_COL]
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\n=== {split_name} ===")
    print(f"Accuracy: {acc:.4f}  |  F1: {f1:.4f}  |  ROC-AUC: {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["not_depressed", "depressed"]))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))

    return {"accuracy": acc, "f1": f1, "roc_auc": auc}


def main():
    train_df, val_df, test_df = load_splits()

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_df[TEXT_COL])
    y_train = train_df[LABEL_COL]

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    evaluate(model, vectorizer, val_df, "Validation")
    evaluate(model, vectorizer, test_df, "Test")

    joblib.dump(model, "baseline_model.joblib")
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
    print("\nSaved baseline_model.joblib and tfidf_vectorizer.joblib")


if __name__ == "__main__":
    main()