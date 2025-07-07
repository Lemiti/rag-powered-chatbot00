# Intelligent Complaint Analysis for Financial Services

A Retrieval-Augmented Generation (RAG) chatbot that transforms customer complaints into actionable insights for CreditTrust Financial.

## 📌 Overview

This project builds an **AI-powered complaint analysis tool** for CreditTrust Financial, enabling internal teams (Product, Support, Compliance) to quickly identify trends in customer feedback. Key features:
- **Semantic Search**: Retrieve relevant complaints using FAISS/ChromaDB.
- **LLM Synthesis**: Generate concise answers with citations using open-source LLMs (e.g., Mistral, Llama 2).
- **User-Friendly UI**: Gradio/Streamlit interface for non-technical users.

## 🛠️ Tech Stack
| Component          | Technology/Tools                          |
|--------------------|------------------------------------------|
| **Data Processing**| Pandas, NumPy, Regex                     |
| **Embeddings**     | SentenceTransformers (`all-MiniLM-L6-v2`)|
| **Vector DB**      | FAISS or ChromaDB                        |
| **LLM**           | Hugging Face Transformers (Mistral/Llama)|
| **RAG Framework**  | LangChain                                |
| **UI**            | Gradio or Streamlit                      |

## 📂 Project Structure
``` 
rag-powered-chatbot00/
├── data/ # Raw and processed datasets
│ ├── consumer_complaints.csv # Original data
│ └── filtered_complaints.csv # Cleaned data (Task 1 output)
├── notebooks/ # EDA and prototyping
│ └── eda_preprocessing.ipynb
├── src/ # Core scripts
│ ├── preprocess.py # Task 1: Data cleaning
│ ├── embed.py # Task 2: Text chunking & embedding
│ ├── rag.py # Task 3: RAG pipeline
│ └── app.py # Task 4: UI implementation
├── vector_store/ # Saved vector DB (Task 2 output)
│
└── README.md # This file
```
