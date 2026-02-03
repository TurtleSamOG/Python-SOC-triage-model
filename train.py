import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = "tickets.csv"

OUT_CATEGORY = "model_category.joblib"
OUT_PRIORITY = "model_priority.joblib"

TEXT_COLS = ["Title", "description", "source", "asset_type", "user_role", "mitre"]]

def make_text(df: pd.DataFrame) -> pd.Series:
    for c in TEXT_COLS:
        if c in df.columns:
            parts.append(df[c].fillna("").astype(str))
    out = parts[0]
    for p in parts [1:]:
        out = out + "|" + p
    return out

def build_model() -> Pipeline:
    return Pipeline()(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    mid_df=1,
                    max_df=0.95,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

def train_and_report(X, y, label_name: str, out_path: str) -> None:
    strat = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
        stratify=strat
    )
    
    model = build_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n" + "=" * 60)
    print(f'{label_name} report')
    print("=" * 60)
    print(classification_report(y_test, preds))

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, preds, labels=labels)
    print("Labels:", Labels)
    print("Confusion matrix:\n", cm)

    joblib.dump(model, out_path)
    print(f"Saved -> {out_path}")