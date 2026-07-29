from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Medicine_Details.csv"
VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "faiss_index"

print("1️⃣ 正在读取并清洗数据...")
df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=['Medicine Name', 'Uses'])
df['Poor Review %'] = pd.to_numeric(df['Poor Review %'], errors='coerce').fillna(0)
df = df[df['Poor Review %'] < 50]

print("2️⃣ 正在把表格转换为精准的 AI 文档块...")
docs = []
for index, row in df.iterrows():
    drug = str(row['Medicine Name']).strip()
    uses = str(row['Uses']).replace('\n', ' ').strip()
    side_effects = str(row['Side_effects']).replace('\n', ' ').strip()

    if side_effects.lower() in ['nan', 'none', '']:
        side_effects = "Consult a pharmacist for details."

    text = f"Medicine: {drug}. Treats: {uses}. Side effects: {side_effects}."
    docs.append(Document(page_content=text))

print(f"✅ 成功提取了 {len(docs)} 种优质药物数据！")

print("3️⃣ 正在唤醒翻译官，转换并建立 FAISS 向量库...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(docs, embeddings)

print("4️⃣ 正在保存数据库...")
vector_db.save_local(str(VECTOR_STORE_PATH))
print("🎉 完美版数据库建立完成！快去 Streamlit 里试试吧！")
