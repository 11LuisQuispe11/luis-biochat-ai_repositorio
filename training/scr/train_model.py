import json
import re
import unicodedata
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "dataset" / "preguntas_luis.csv"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    """
    Normaliza texto para reducir ruido:
    - minúsculas
    - sin tildes
    - sin signos raros
    - espacios limpios
    """
    text = str(text).lower().strip()

    text = unicodedata.normalize("NFD", text)
    text = "".join(
        character
        for character in text
        if unicodedata.category(character) != "Mn"
    )

    text = re.sub(r"[^a-z0-9ñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)

    if "text" not in df.columns or "intent" not in df.columns:
        raise ValueError("El dataset debe tener las columnas: text,intent")

    df = df.dropna(subset=["text", "intent"])
    df["text_normalized"] = df["text"].apply(normalize_text)

    return df


def train_model(df: pd.DataFrame) -> Pipeline:
    X = df["text_normalized"]
    y = df["intent"]

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True
                )
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    model.fit(X, y)

    return model


def evaluate_model(model: Pipeline, df: pd.DataFrame) -> dict:
    X = df["text_normalized"]
    y_true = df["intent"]

    y_pred = model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)

    print("\n==============================")
    print("EVALUACIÓN DEL MODELO")
    print("==============================")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_true, y_pred))

    metrics = {
        "accuracy": accuracy,
        "total_samples": len(df),
        "total_classes": int(df["intent"].nunique()),
        "classes": sorted(df["intent"].unique().tolist())
    }

    return metrics


def save_model(model: Pipeline, metrics: dict) -> None:
    model_path = MODELS_DIR / "intent_classifier.pkl"
    metrics_path = MODELS_DIR / "metrics.json"

    joblib.dump(model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print("\n==============================")
    print("ARTEFACTOS GENERADOS")
    print("==============================")
    print(f"Modelo guardado en: {model_path}")
    print(f"Métricas guardadas en: {metrics_path}")


def main():
    print("Cargando dataset...")
    df = load_dataset()

    print(f"Total de registros: {len(df)}")
    print(f"Total de intenciones: {df['intent'].nunique()}")

    print("\nDistribución de clases:")
    print(df["intent"].value_counts())

    print("\nEntrenando modelo...")
    model = train_model(df)

    metrics = evaluate_model(model, df)

    save_model(model, metrics)


if __name__ == "__main__":
    main()