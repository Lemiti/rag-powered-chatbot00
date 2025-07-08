import os
import pandas as pd
import numpy as np
import pickle
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_data(csv_path="/content/drive/MyDrive/10Acadamy/data/processed/filtered_compliants.csv"):
    df = pd.read_csv(csv_path)
    return df


def chunk_texts(df, chunk_size=300, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []
    metadata = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        chunks = text_splitter.split_text(row["clean_narrative"])
        all_chunks.extend(chunks)
        metadata.extend([{"product": row["Product"], "source_index": i}] * len(chunks))

    return all_chunks, metadata


def embed_chunks(chunks, model_name="all-MiniLM-L6-v2", batch_size=128):
    model = SentenceTransformer(model_name)
    all_embeddings = []

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
        batch = chunks[i:i + batch_size]
        embeddings = model.encode(batch)
        all_embeddings.extend(embeddings)

    return np.array(all_embeddings)


def save_faiss_index(embeddings, save_dir="/content/drive/MyDrive/10Acadamy/rag-powered-chatbot00/vector_store"):
    os.makedirs(save_dir, exist_ok=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(save_dir, "faiss_index.bin"))


def save_metadata(chunks, metadata, save_dir="/content/drive/MyDrive/10Acadamy/rag-powered-chatbot00/vector_store"):
    with open(os.path.join(save_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)


if __name__ == "__main__":
    print("🔹 Loading data...")
    df = load_data()

    print("🔹 Splitting narratives into chunks...")
    chunks, metadata = chunk_texts(df)

    print(f"🔹 Total chunks created: {len(chunks)}")

    print("🔹 Generating embeddings...")
    embeddings = embed_chunks(chunks)

    print("🔹 Saving FAISS index and metadata...")
    save_faiss_index(embeddings)
    save_metadata(chunks, metadata)

    print("✅ All done! Vector store ready in '/content/drive/MyDrive/10Acadamy/rag-powered-chattbot00/vector_store/'")
