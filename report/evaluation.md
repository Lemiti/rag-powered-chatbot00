# 🧠 Week 6 – RAG Chatbot Evaluation Report

This evaluation assesses the performance of the Retrieval-Augmented Generation (RAG) system built using FAISS and Falcon-7B-Instruct (via `transformers`) on real consumer complaint data.

## ✅ Setup Overview

- **Model**: `tiiuae/falcon-7b-instruct` (locally via Transformers)
- **Retrieval**: FAISS, top-5 chunk similarity
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Prompt Style**: Few-shot summarization with citation guidance
- **Query Format**: Natural language questions from the user

---

## 🧪 Evaluation Table

| # | Question | Answer Summary | Sources Used | Rating (1–5) | Comments |
|---|----------|----------------|--------------|---------------|----------|
| 1 | What are customers saying about late fees? | Users frequently mention unfair or hidden fees, especially when billing is delayed. Most complaints express dissatisfaction and confusion. | ✅ | 4.5 | Answer is accurate and reflects diverse chunks, could be slightly more specific. |
| 2 | Why are people unhappy with credit cards? | Complaints focus on excessive fees, poor customer support, and privacy concerns around account changes and interest rates. | ✅ | 5 | Excellent — matches tone and detail of original data. |
| 3 | How is the customer service experience? | Described as "poor" and "the worst ever" — with reports of being ignored or redirected without help. | ✅ | 5 | Precise and well-supported by retrieved chunks. |
| 4 | What issues exist with money transfers? | Users report delays, lack of availability, and poor communication around processing or failures. | ✅ | 4.5 | Answer is good but could be more structured (e.g. grouped issues). |
| 5 | Are there complaints about fraud? | Yes — several complaints highlight unauthorized transactions and poor fraud handling processes. | ✅ | 5 | Clear, direct, and supported by all chunks. |

---

## 🔍 Overall Observations

- **Contextual grounding**: The model reliably incorporates relevant complaint excerpts.
- **Factual accuracy**: High alignment between answer content and chunk data.
- **Citations**: While citations were not numbered in output, the structure allows them and the prompt encourages them.
- **Response style**: Informative and fluent, appropriate for a financial assistant.

---

## 🔧 Areas for Improvement

- Add explicit source number references like `[1], [2]` to further enhance traceability.
- Improve formatting and summarization consistency (e.g., bullets or grouping issues).

---

## ✅ Final Rating

**Overall Effectiveness**: **4.7 / 5**

The system performs well in generating informative, grounded, and relevant answers to real-world questions. It is production-ready for demo purposes and serves as a solid base for future enhancements.

---


