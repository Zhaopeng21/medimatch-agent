from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "nz_medicine_faiss"


@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        db = FAISS.load_local(str(VECTOR_STORE_PATH), embeddings, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        return None


vector_db = load_vector_db()
