import json
import ast
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

ENTITIES_FILE = Path("output/erpnext_entities.json")
INDEX_FILE = Path("output/erpnext.index")
META_FILE = Path("output/metadata.json")

model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI()

def build_documents():
    data = json.loads(ENTITIES_FILE.read_text(encoding="utf-8"))
    documents = []

    for file, entities in data.items():
        for ent in entities:
            if ent["type"] == "class":
                documents.append({
                    "text": ent["source_code"],
                    "metadata": {
                        "file": file,
                        "name": ent["name"],
                        "type": "class"
                    }
                })
                for m in ent["methods"]:
                    documents.append({
                        "text": m["source_code"],
                        "metadata": {
                            "file": file,
                            "class": ent["name"],
                            "name": m["name"],
                            "type": "method"
                        }
                    })
            else:
                documents.append({
                    "text": ent["source_code"],
                    "metadata": {
                        "file": file,
                        "name": ent["name"],
                        "type": "function"
                    }
                })

    return documents

def build_index():
    print("📦 Building documents...")
    docs = build_documents()

    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    INDEX_FILE.parent.mkdir(exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_text(json.dumps(docs, indent=2))

    print("✅ Vector index built")
    print(f"Chunks stored: {len(docs)}")

def retrieve(query, top_k=5):
    index = faiss.read_index(str(INDEX_FILE))
    metadata = json.loads(META_FILE.read_text())

    q_emb = model.encode([query]).astype("float32")
    _, indices = index.search(q_emb, top_k)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])
    return results

def ask(question):
    docs = retrieve(question)

    context = "\n\n".join([d["text"] for d in docs])

    prompt = f"""
You are an ERPNext code assistant.
Answer only using the provided code context.

Context:
{context}

Question:
{question}

Answer clearly:
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    # Step A — Build index if not exists
    if not INDEX_FILE.exists():
        build_index()

    print("\n🧠 ERPNext RAG Ready!")
    print("Type your questions about ERPNext code.\n")

    while True:
        q = input("Ask ERPNext > ")
        if q.lower() in ["exit", "quit"]:
            break
        print("\n" + ask(q) + "\n")