"""
Data loading + preprocessing for the depression-detection project.

Dataset: infamouscoder/depression-reddit-cleaned (Kaggle)
Columns: clean_text (str), is_depression (0/1)

Usage:
    1. Download the dataset (via Kaggle CLI or kagglehub) so you have a CSV
       locally, e.g. "depression_dataset_reddit_cleaned.csv".
    2. Run this script, or import `load_splits()` from a notebook.

    # Kaggle CLI:
    #   kaggle datasets download -d infamouscoder/depression-reddit-cleaned
    #   unzip depression-reddit-cleaned.zip

    # or with kagglehub:
    #   import kagglehub
    #   path = kagglehub.dataset_download("infamouscoder/depression-reddit-cleaned")
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_CSV_PATH = "depression_dataset_reddit_cleaned.csv"
TEXT_COL = "clean_text"
LABEL_COL = "is_depression"

RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # taken out of the remaining train split


def load_raw(path: str = RAW_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert TEXT_COL in df.columns and LABEL_COL in df.columns, (
        f"Expected columns '{TEXT_COL}' and '{LABEL_COL}', got {list(df.columns)}"
    )
    return df


def basic_clean(text: str) -> str:
    """Light normalization on top of the dataset's existing cleaning."""
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # strip URLs
    text = re.sub(r"\s+", " ", text)                # collapse whitespace
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df[TEXT_COL] = df[TEXT_COL].apply(basic_clean)
    df = df[df[TEXT_COL].str.len() > 0]
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df = df.drop_duplicates(subset=[TEXT_COL])
    return df.reset_index(drop=True)


def load_splits(path: str = RAW_CSV_PATH):
    """Returns (train_df, val_df, test_df), stratified on the label."""
    df = preprocess(load_raw(path))

    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[LABEL_COL], random_state=RANDOM_SEED
    )
    train, val = train_test_split(
        train_val, test_size=VAL_SIZE, stratify=train_val[LABEL_COL], random_state=RANDOM_SEED
    )

    for name, split in [("train", train), ("val", val), ("test", test)]:
        pos_rate = split[LABEL_COL].mean()
        print(f"{name:5s}: {len(split):5d} rows | positive rate: {pos_rate:.3f}")

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    train_df, val_df, test_df = load_splits()

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)
    print("Saved train.csv, val.csv, test.csv")