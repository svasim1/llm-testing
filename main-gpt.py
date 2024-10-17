import os
# Custom cache
PATH = 'D:/dev/projekt/llm-testing/.cache'
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model_name = "AI-Sweden-Models/gpt-sw3-1.3b-instruct"
device = 0 if torch.cuda.is_available() else -1
prompt_query = input(f"\nEnter your query: ")

# Load the GPT-3 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline('text-generation', model=AutoModelForCausalLM.from_pretrained(model_name), tokenizer=tokenizer, device=device)

# Embedder model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Context (should be documents in ./data/)



documents = []

for line in open('./data/lagbok.txt', 'r'):
    documents.append(line)

# Embed the context (./data/)
embeddings = embedder.encode(documents, convert_to_tensor=False)

# ????
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Encode Query and Retrieve Relevant Documents
query_embedding = embedder.encode([prompt_query])
top_k = 3  # Number of relevant documents to retrieve
distances, indices = index.search(np.array(query_embedding), top_k)

# Retrieve the top matching documents
retrieved_docs = "\n".join([documents[idx] for idx in indices[0]])

# Combine Retrieved Documents with Query for Generation
prompt = f"Relevant information:\n{retrieved_docs}\n\nUser: {prompt_query}\nBot:"

# Generate Response
response = generator(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=1)[0]["generated_text"]
print(response)
