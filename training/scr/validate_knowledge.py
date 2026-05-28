import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = BASE_DIR / "dataset" / "preguntas_luis.csv"
KNOWLEDGE_PATH = BASE_DIR / "knowledge" / "biografia_luis.json"


def main():
    df = pd.read_csv(DATASET_PATH)

    if "intent" not in df.columns:
        raise ValueError("El dataset debe tener una columna llamada 'intent'.")

    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as file:
        knowledge = json.load(file)

    dataset_intents = set(df["intent"].dropna().unique())
    knowledge_keys = set(knowledge.keys())

    missing_in_json = dataset_intents - knowledge_keys
    unused_in_dataset = knowledge_keys - dataset_intents

    print("\n==============================")
    print("VALIDACIÓN DATASET VS JSON")
    print("==============================")

    print(f"Intenciones en dataset: {len(dataset_intents)}")
    print(f"Claves en JSON: {len(knowledge_keys)}")

    if missing_in_json:
        print("\n❌ Intenciones del dataset que NO existen en el JSON:")
        for intent in sorted(missing_in_json):
            print(f"- {intent}")
    else:
        print("\n✅ Todas las intenciones del dataset existen en el JSON.")

    if unused_in_dataset:
        print("\n⚠️ Claves del JSON que todavía NO tienen preguntas en el dataset:")
        for key in sorted(unused_in_dataset):
            print(f"- {key}")
    else:
        print("\n✅ Todas las claves del JSON tienen preguntas en el dataset.")

    print("\nValidación finalizada.")


if __name__ == "__main__":
    main()