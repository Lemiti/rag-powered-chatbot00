import faiss
import os
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class FinancialComplaintAnalyzer:
    def __init__(self):
        self.index, self.chunks, self.metadata, self.embedder = self._load_assets()
        self.llm_pipeline = self._initialize_llm()
        
    def _load_assets(self):
        """Load vector store and embedding model"""
        try:
            index = faiss.read_index("vector_store/faiss_index.bin")
            with open("vector_store/chunks.pkl", "rb") as f:
                chunks = pickle.load(f)
            with open("vector_store/metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            return index, chunks, metadata, embedder
        except Exception as e:
            raise RuntimeError(f"Failed to load assets: {e}")

    def _initialize_llm(self):
        """Initialize the LLM pipeline with optimized settings"""
        try:
            model = "tiiuae/falcon-7b-instruct"
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                truncation=True,
                padding=True
            )
            return pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens=300,  # Reduced for more focused answers
                do_sample=True,
                top_p=0.9,
                temperature=0.3
            )
        except Exception as e:
            raise RuntimeError(f"LLM initialization failed: {e}")

    def retrieve_complaints(self, query, top_k=5):
        """Retrieve relevant complaints with improved error handling"""
        try:
            q_embed = self.embedder.encode([query])
            distances, indices = self.index.search(np.array(q_embed), top_k)
            return (
                [self.chunks[i] for i in indices[0]],
                [self.metadata[i] for i in indices[0]]
            )
        except Exception as e:
            raise RuntimeError(f"Retrieval failed: {e}")

    def generate_answer(self, query, retrieved_chunks):
        """Generate answer with better prompt engineering"""
        try:
            context = "\n".join([
                f"[Excerpt {i+1}]: {chunk}" 
                for i, chunk in enumerate(retrieved_chunks)
            ])
            
            prompt = f"""As a financial complaints analyst, provide a detailed answer using ONLY these customer complaint excerpts:

{context}

Question: {query}

Guidelines:
1. Identify 3-5 key issues mentioned
2. Cite specific excerpts like [1], [2] for each point
3. If no relevant info, state "No relevant complaints found"
4. Keep response under 150 words

Analysis:"""
            
            result = self.llm_pipeline(
                prompt,
                return_full_text=False,
                eos_token_id=self.llm_pipeline.tokenizer.eos_token_id
            )
            
            return result[0]['generated_text'].strip()
        except Exception as e:
            return f"Analysis failed: {e}"

    def analyze_query(self, query, top_k=5):
        """Complete RAG pipeline with enhanced output"""
        try:
            chunks, meta = self.retrieve_complaints(query, top_k)
            if not chunks:
                return {
                    "answer": "No relevant complaints found",
                    "sources": []
                }
                
            answer = self.generate_answer(query, chunks)
            return {
                "question": query,
                "answer": answer,
                "sources": [
                    {
                        "product": m.get('product', 'Unknown'),
                        "excerpt": c[:200] + "..."
                    } for m, c in zip(meta, chunks)
                ]
            }
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    analyzer = FinancialComplaintAnalyzer()
    
    test_questions = [
        "What are the most common complaints about late fees?",
        "What specific issues do customers report with credit card services?",
        "How do customers describe their customer service experiences?",
        "What problems occur with money transfer services?",
        "What fraud-related complaints have been filed?"
    ]

    for question in test_questions:
        try:
            result = analyzer.analyze_query(question)
            
            print(f"\n{'='*50}\nQuestion: {result['question']}\n{'-'*50}")
            print("Analysis:\n", result['answer'])
            
            if result.get('sources'):
                print("\nSupporting Excerpts:")
                for i, source in enumerate(result['sources']):
                    print(f"[{i+1}] {source['product']}: {source['excerpt']}")
                    
        except Exception as e:
            print(f"Error processing question: {e}")