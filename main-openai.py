from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load the API key from the .env file
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Store the index
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Create retriever from index
retriever = index.as_retriever()

# Inoital prompt that gives context to the AI
context_prompt = """
Du är en AI-assistent som hjälper till att sammanfatta lagboken på svenska.
Var alltid artig, använd formellt språk och ge detaljerade svar. 
Om användaren ställer en fråga som inte är relaterad till lagboken, påminn dem vänligt om att du endast kan hjälpa till med lagboken.
Om assistenten inte vet svaret på en fråga, ska den ärligt säga att den inte vet svaret.

Här är relevanta dokument för sammanfattningen av lagboken:
{context_str}

Instruktion: Baserat på dokumenten ovan, ge ett detaljerat svar på användarfrågan nedan.
Var noggran och svara "vet ej" om svaret inte finns i dokumentet. Utgå att allting som inte står i dokumentet är felaktigt, och du vill inte förmedla felaktik information.
"""

# Prompt for condensing the conversation
condense_prompt = """
Givet följande konversation mellan en användare och en AI-assistent samt en uppföljningsfråga från användaren,
omformulera uppföljningsfrågan så att den blir en fristående fråga.

Chatthistorik:
{chat_history}
Uppföljningsfråga: {question}
Fristående fråga:
"""




# Hardcoded chat history to start the conversation
custom_chat_history = [
    ChatMessage(
        role=MessageRole.USER,
        content="Hej assistenten, idag sammarfattar vi lagboken på svenska.",
    ),
    ChatMessage(role=MessageRole.ASSISTANT, content="Okej, låter bra."),
]

# Query the data
query_engine = index.as_query_engine()
chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=retriever,
    memory = ChatMemoryBuffer.from_defaults(token_limit=3900),
    query_engine=query_engine,
    context_prompt=context_prompt,
    condense_prompt=condense_prompt,
    chat_history=custom_chat_history,
    verbose=True,
)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class Question(BaseModel):
    question: str    

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
def chat(input: Question):
    response = run_your_chatbot(input.question)
    return {"response": response}

# Function to run chatbot and return message
def run_your_chatbot(message):
    response = chat_engine.chat(message)
    return response

# Chat endpoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)