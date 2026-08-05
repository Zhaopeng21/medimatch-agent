"""Build the local FAISS medicine index from the project's existing CSV."""

from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "pipeline" / "legacy" / "Medicine_Details.csv"
VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "archive" / "legacy_faiss_index"


def clean_value(value: object) -> str:
    text = str(value).replace("\n", " ").strip()
    return "" if text.lower() in {"nan", "none"} else text


print("Reading and cleaning medicine data...")
df = pd.read_csv(DATA_PATH).dropna(subset=["Medicine Name", "Uses"])
df["Poor Review %"] = pd.to_numeric(df["Poor Review %"], errors="coerce").fillna(0)
df = df[df["Poor Review %"] < 50]

print("Building medicine documents...")
docs = []
for _, row in df.iterrows():
    medicine_name = clean_value(row["Medicine Name"])
    treats = clean_value(row["Uses"])
    composition = clean_value(row.get("Composition", ""))
    side_effects = clean_value(row.get("Side_effects", "")) or "Consult a pharmacist for details."
    docs.append(
        Document(
            page_content=(
                f"Medicine: {medicine_name}. Treats: {treats}. "
                f"Composition: {composition}. Side effects: {side_effects}."
            ),
            metadata={
                "medicine_name": medicine_name,
                "treats": treats,
                "composition": composition,
                "side_effects": side_effects,
            },
        )
    )

print(f"Prepared {len(docs)} medicine documents.")
print("Building FAISS vector index...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
FAISS.from_documents(docs, embeddings).save_local(str(VECTOR_STORE_PATH))
print("FAISS index build complete.")
