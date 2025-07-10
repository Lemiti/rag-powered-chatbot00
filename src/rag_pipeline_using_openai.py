import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI  # Updated import for modern LangChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Suppress TensorFlow/CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def load_assets():
    """Load FAISS index, chunks, and metadata"""
    try:
        index = faiss.read_index("../vector_store/faiss_index.bin")  # Use relative path
        with open("../vector_store/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        with open("../vector_store/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return index, chunks, metadata, embedder
    except Exception as e:
        print(f"Error loading assets: {e}")
        raise

def retrieve_chunks(query, index, chunks, metadata, embedder, top_k=5):
    """Semantic search for relevant chunks"""
    q_embed = embedder.encode([query])
    distances, indices = index.search(np.array(q_embed), top_k)
    return [chunks[i] for i in indices[0]], [metadata[i] for i in indices[0]]

def generate_answer(query, retrieved_chunks, api_key):
    """Generate answer using OpenAI"""
    context = "\n\n".join(retrieved_chunks)
    
    prompt_template = """You are a financial analyst assistant. Use ONLY these complaints:
    
    {context}
    
    Question: {question}
    Answer concisely and cite sources like [1],[2]:"""
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
    
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo",
        openai_api_key=api_key  # Pass API key explicitly
    )
    return LLMChain(llm=llm, prompt=prompt).run({"context": context, "question": query})

def answer_query(query, api_key, top_k=5):
    """End-to-end RAG pipeline"""
    index, chunks, metadata, embedder = load_assets()
    retrieved, meta = retrieve_chunks(query, index, chunks, metadata, embedder, top_k)
    answer = generate_answer(query, retrieved, api_key)
    return answer, retrieved, meta

if __name__ == "__main__":
    # Initialize with your OpenAI API key
    OPENAI_API_KEY = "sk-..."  # Replace with your actual key
    
    question = "What are common issues with buy now pay later services?"
    try:
        answer, context, sources = answer_query(question, OPENAI_API_KEY)
        
        print(f"\nQuestion:\n{question}\n")
        print("Answer:\n", answer)
        print("\nSources:")
        for i, (src, txt) in enumerate(zip(sources, context)):
            print(f"[{i+1}] {src['product']}: {txt[:150]}...")
    except Exception as e:
        print(f"Error: {e}")