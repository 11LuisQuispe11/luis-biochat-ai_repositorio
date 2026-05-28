import re
import unicodedata
from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "intent_classifier.pkl"


def normalize_text(text: str) -> str:
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


def predict_question(model, question: str) -> None:
    question_normalized = normalize_text(question)

    predicted_intent = model.predict([question_normalized])[0]

    probabilities = model.predict_proba([question_normalized])[0]
    classes = model.classes_

    ranking = sorted(
        zip(classes, probabilities),
        key=lambda item: item[1],
        reverse=True
    )

    print("\nPregunta:", question)
    print("Texto normalizado:", question_normalized)
    print("Intención predicha:", predicted_intent)
    print("\nTop probabilidades:")

    for intent, probability in ranking[:5]:
        print(f"- {intent}: {probability:.4f}")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No existe el modelo. Primero ejecuta: python src/train_model.py"
        )

    model = joblib.load(MODEL_PATH)

    test_questions = [
        "¿Luis sabe trabajar con Python y SQL?",
        "¿Ha trabajado en bancos?",
        "¿Qué experiencia tiene en minería?",
        "¿Por qué debería contratarlo?",
        "¿Cómo puedo contactar a Luis?",
        "¿Qué proyectos de automatización tiene?"
    ]

    for question in test_questions:
        predict_question(model, question)


if __name__ == "__main__":
    main()