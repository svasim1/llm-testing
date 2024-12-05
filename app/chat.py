import os
from openai import OpenAI
from fastapi import HTTPException
from logging_conf import logger
import re
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage

# Constants
PERSIST_DIR = "storage"

token_usage_stats = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
}

# Initial prompt that gives context to the AI
context_prompt = """
Du är en AI-assistent som hjälper till att sammanfatta lagboken på svenska.
Var alltid artig, använd formellt språk och ge detaljerade svar. 
Om användaren ställer en fråga som inte är relaterad till lagboken, påminn dem vänligt om att du endast kan hjälpa till med lagboken.
Om assistenten inte vet svaret på en fråga, ska den ärligt säga att den inte vet svaret.

Här är relevanta dokument för sammanfattningen av lagboken:
{context_str}

Instruktion: Baserat på dokumenten ovan, ge ett detaljerat svar på användarfrågan nedan.
Var noggran och svara "vet ej" om svaret inte finns i dokumentet. Utgå att allting som inte står i dokumentet är felaktigt, och du vill inte förmedla felaktig information.
Ditt svar kommer visat på en hemsida, och använd därför inte till exempel markdown formatering. Svara inte för utförligt och se till att all information får plats i svaret.
"""

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Split the text into chunks using sentences
def split_text_into_chunks(text, chunk_size=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    for sentence in sentences:
        if sum(len(s) for s in current_chunk) + len(sentence) <= chunk_size:
            current_chunk.append(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Read the lagbok.txt file
with open("../data/lagbok.txt", "r", encoding="utf-8") as file:
    lagbok_text = file.read()

# Split the text into chunks
chunks = split_text_into_chunks(lagbok_text)

# Convert chunks to Document objects
documents = [Document(text=chunk) for chunk in chunks] 

# Define your persist directory path
try:
    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR)
        # Create the index
        index = VectorStoreIndex.from_documents(documents)
        # Store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        logger.debug(f"Index stored in {PERSIST_DIR}")
    else:
        # Load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        logger.debug(f"Index loaded from {PERSIST_DIR}")
except Exception as e:
    logger.error(f"Error while setting up the index: {e}")
    raise

# Create retriever from index
retriever = index.as_retriever()

# Function to moderate content
async def moderate_content(content: str):
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=content,
    )
    if response.results[0].flagged:
        raise HTTPException(status_code=400, detail="Content flagged as unsafe")

# Function to run chatbot and return message and sources
async def chatbot(message, user_id):
    logger.info(f"User ID: {user_id}")
    logger.info(f"Message: {message}")
    try:
        await moderate_content(message)
        
        # Retrieve relevant documents
        results = retriever.retrieve(message)
        sources = [result.node.get_content() for result in results]
        
        # Prepare the context with the retrieved documents
        context = "\n\n".join(sources)
        
        # Make API call to OpenAI with user ID and context
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": context_prompt},
                {"role": "user", "content": message},
                {"role": "system", "content": f"Svara endast efter dessa relevanta källor och hänvisa alltid utifrån kapitel: {sources}"}
            ],
            max_tokens=500, # The higher the number, the longer the response (uses more tokens :p)
            stop=["###"],
            user=user_id
        )
        
        # Extract the actual content from the response
        response_content = response.choices[0].message.content
        logger.debug(f"Response content: {response_content}")
        
        # Log token usage details
        global token_usage_stats
        token_usage_stats["prompt_tokens"] += response.usage.prompt_tokens
        token_usage_stats["completion_tokens"] += response.usage.completion_tokens
        token_usage_stats["total_tokens"] += response.usage.total_tokens
        
        return response_content, sources
    except Exception as e:
        logger.error(f"Error in chatbot: {e}")
        raise