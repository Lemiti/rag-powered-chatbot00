import faiss
import os
import pickle
import numpy as np
import torch
import traceback
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline

# Suppress TensorFlow/CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_assets():
    """Load FAISS index, chunks, and metadata"""
    try:
        index = faiss.read_index("vector_store/faiss_index.bin")
        with open("vector_store/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        with open("vector_store/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return index, chunks, metadata, embedder
    except Exception as e:
        print(f"Error loading assets: {e}")
        raise

def retrieve_chunks(query, index, chunks, metadata, embedder, top_k=5):
    """Semantic search for relevant chunks"""
    try:
        q_embed = embedder.encode([query])
        distances, indices = index.search(np.array(q_embed), top_k)
        return [chunks[i] for i in indices[0]], [metadata[i] for i in indices[0]]
    except Exception as e:
        print(f"Retrieval error: {e}")
        return [], []

def initialize_falcon_llm():
    """Initialize Falcon-7B-Instruct pipeline"""
    model = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model)
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

def generate_answer(query, retrieved_chunks, llm_pipeline):
    """Generate answer using Falcon-7B-Instruct"""
    try:
        context = "\n".join([f"[Source {i+1}] {chunk}" for i, chunk in enumerate(retrieved_chunks)])
        
        prompt = f"""As a financial analyst, answer the question using ONLY the provided complaint excerpts.
        
Complaint Excerpts:
{context}

Question: {query}

Answer concisely and cite sources like [1],[2] where applicable. If no relevant information is found, state "No relevant complaints found."

Answer:"""
        
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
        sequences = llm_pipeline(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            return_full_text=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        return sequences[0]['generated_text'].strip()
    except Exception as e:
        return f"Error generating answer: {e}\n{traceback.format_exc()}"

def answer_query(query, llm_pipeline, top_k=5):
    """End-to-end RAG pipeline"""
    try:
        index, chunks, metadata, embedder = load_assets()
        retrieved, meta = retrieve_chunks(query, index, chunks, metadata, embedder, top_k)
        
        if not retrieved:
            return "No relevant complaints found", [], []
            
        answer = generate_answer(query, retrieved, llm_pipeline)
        return answer, retrieved, meta
    except Exception as e:
        return f"Pipeline error: {e}", [], []

if __name__ == "__main__":
    # Initialize components
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    llm_pipeline = initialize_falcon_llm()
    
    test_questions = [
        "What are customers saying about late fees?",
        "Why are people unhappy with credit cards?",
        "How is the customer service experience?",
        "What issues exist with money transfers?",
        "Are there complaints about fraud?"
    ]

    for question in test_questions:
        try:
            answer, context, sources = answer_query(question, llm_pipeline)
            
            print(f"\nQuestion:\n{question}\n")
            print("Answer:\n", answer)
            
            if sources:
                print("\nSources:")
                for i, (src, txt) in enumerate(zip(sources, context)):
                    print(f"[{i+1}] {src.get('product','Unknown')}: {txt[:150]}...")
                    
        except Exception as e:
            print(f"Error processing question '{question}': {e}")