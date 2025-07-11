import gradio as gr
from rag_pipeline import answer_query, initialize_falcon_llm

#Load model only once
llm_pipeline = initialize_falcon_llm()

def chatbot_response(query):
  answer, chunks, sources = answer_query(query, llm_pipeline)


  #Format sources
  source_texts = ""
  for i, (src, chunk) in enumerate(zip(sources, chunks)):
    product = src.get("product", "Unknown")
    preview = chunk[:150].strip().replace("\n", " ")
    source_texts += f"[{i+1}] {product}: {preview}...\n"

  return f"📌 **Answer:**\n{answer}\n\n📚 **Source:**\n{source_texts}"


#Launch Gradio Interface
gr.Interface(
  fn=chatbot_response,
  inputs=gr.Textbox(label="Ask a financial question", placeholder="e.g., What are customers saying about late fees?"),
  outputs=gr.Markdown(label="Answer with Sources"),
  title="💬 Financial Compliants Chatbot",
  description="Powered by FAISS + Falcon-7B-Instruct + Transformers"
).launch(share=True)