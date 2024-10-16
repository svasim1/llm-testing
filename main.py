import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model_name = "AI-Sweden-Models/gpt-sw3-126m-instruct"
device = 0 if torch.cuda.is_available() else -1
prompt_query = input("Enter your query: ")

# Load the GPT-3 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline('text-generation', model=AutoModelForCausalLM.from_pretrained(model_name), tokenizer=tokenizer, device=device)

# Embedder model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Context (should be documents in ./data/)
documents = [
    "Träd är viktiga för att de ger syre.",
    "Skogar hjälper till att reglera klimatet.",
    "Fotosyntesen är processen genom vilken växter tillverkar sin mat."
    "Kontrakt är en överenskommelse mellan två eller flera parter som skapar en rättslig skyldighet att göra eller inte göra något."
]

# Embed the context (./data/)
embeddings = embedder.encode(documents, convert_to_tensor=False)

# ????
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Step 6: Encode Query and Retrieve Relevant Documents
query_embedding = embedder.encode([prompt_query])
top_k = 3  # Number of relevant documents to retrieve
distances, indices = index.search(np.array(query_embedding), top_k)

# Retrieve the top matching documents
retrieved_docs = "\n".join([documents[idx] for idx in indices[0]])

# Step 7: Combine Retrieved Documents with Query for Generation
prompt = f"Relevant information:\n{retrieved_docs}\n\nUser: {prompt_query}\nBot:"

# Step 8: Generate Response
response = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.6, top_p=1)[0]["generated_text"]
print(response)
