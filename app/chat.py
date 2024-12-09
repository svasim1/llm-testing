import os
from openai import OpenAI
from fastapi import HTTPException
from logging_conf import logger
from data_processing import create_or_load_index

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

# Create or load the index
index = create_or_load_index()

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
async def chatbot(message, user_email):
    logger.info(f"User Email: {user_email}")
    logger.info(f"Message: {message}")
    try:
        await moderate_content(message)
        
        # Retrieve relevant documents
        results = retriever.retrieve(message)
        sources = [result.node.get_content() for result in results]
        
        # Prepare the context with the retrieved documents
        context = "\n\n".join(sources)
        
        # Make API call to OpenAI with user email and context
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": context_prompt},
                {"role": "user", "content": message},
                {"role": "system", "content": f"Svara endast efter dessa relevanta källor och hänvisa alltid utifrån kapitel: {sources}"}
            ],
            max_tokens=500, # The higher the number, the longer the response (uses more tokens :p)
            stop=["###"],
            user=user_email
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
        raise HTTPException(status_code=500, detail="Internal Server Error")