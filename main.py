from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
import html

# Ladda en ny modell via Hugging Face
model_id = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# Använd Hugging Face's pipeline för embeddings med BAAI/bge-base-en-v1.5
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Skapa en "fake lagbok" med fiktiva lagtexter
with open("data/lagbok.txt", "r", encoding="utf-8") as f:
    fake_lagbok = f.readlines()

# Skapa en vektorbutik med den fiktiva lagboken
vector_store = FAISS.from_texts(fake_lagbok, embedder)

# Skapa en RetrievalQA-kedja med embeddings
llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text-generation",
    pipeline_kwargs={"temperature": 0.7, "max_new_tokens": 100, "do_sample": True}
)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

# Ställ en fråga och få ett svar
query = input("Query: ")
result = qa_chain.invoke(query)

# Format the result as HTML
html_output = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QA Result</title>
</head>
<body>
    <h1>Query</h1>
    <p>{html.escape(query)}</p>
    <h1>Result</h1>
    <p>{html.escape(result['result'])}</p>
</body>
</html>
"""

# Save the HTML to a file
with open("result.html", "w", encoding="utf-8") as file:
    file.write(html_output)

print("Result saved to result.html")