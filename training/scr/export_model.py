import json
from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "intent_classifier.pkl"
OUTPUT_PATH = BASE_DIR / "models" / "intent_model_web.json"


def get_pipeline_step(model, possible_names):
    """
    Busca un paso dentro del Pipeline usando varios nombres posibles.
    Esto evita errores si en train_model.py usamos 'classifier' o 'clf'.
    """
    for name in possible_names:
        if name in model.named_steps:
            return model.named_steps[name]

    available_steps = list(model.named_steps.keys())

    raise KeyError(
        f"No se encontró ninguno de estos pasos: {possible_names}. "
        f"Pasos disponibles en el modelo: {available_steps}"
    )


def export_model_to_json():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe el modelo en: {MODEL_PATH}. "
            "Primero ejecuta train_model.py"
        )

    model = joblib.load(MODEL_PATH)

    tfidf = get_pipeline_step(model, ["tfidf", "vectorizer"])
    classifier = get_pipeline_step(model, ["classifier", "clf", "model"])

    payload = {
        "model_type": "tfidf_logistic_regression",
        "description": "Modelo NLP para clasificar intenciones de preguntas sobre Luis Quispe Inquil.",
        "classes": classifier.classes_.tolist(),
        "vocabulary": tfidf.vocabulary_,
        "idf": tfidf.idf_.tolist(),
        "coefficients": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
        "ngram_range": list(tfidf.ngram_range),
        "normalization": {
            "lowercase": True,
            "remove_accents": True,
            "remove_symbols": True
        }
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print("\n==============================")
    print("MODELO EXPORTADO PARA WEB")
    print("==============================")
    print(f"Modelo Python leído desde: {MODEL_PATH}")
    print(f"Modelo web generado en: {OUTPUT_PATH}")
    print(f"Total de clases/intenciones: {len(payload['classes'])}")
    print(f"Total de términos en vocabulario: {len(payload['vocabulary'])}")

    print("\nClases exportadas:")
    for class_name in payload["classes"]:
        print(f"- {class_name}")


if __name__ == "__main__":
    export_model_to_json()