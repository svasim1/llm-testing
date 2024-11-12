from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field

# Setup logging
log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_directory, log_filename), encoding='utf-8'),
        #logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load the environment variables - see .env.sample
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PERSIST_DIR = os.getenv('PERSIST_DIR')
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS')

if not OPENAI_API_KEY or not PERSIST_DIR or not ALLOWED_ORIGINS:
    logger.error("Required environment variables are not set")
    raise EnvironmentError("Required environment variables are not set")

# Store the index
try:
    if not os.path.exists(PERSIST_DIR):
        # load the documents and create the index
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        # store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        logger.info(f"Index stored in ${PERSIST_DIR}")
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        logger.info(f"Index loaded from ${PERSIST_DIR}")
except Exception as e:
    logger.error(f"Error while setting up the index: {e}")
    raise

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
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Setup FastAPI
app = FastAPI()
app.state.limiter = limiter

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, pattern=r'^[a-zA-Z0-9\s\.,?!åäöÅÄÖ":;]+$')

@app.post("/chat")
@limiter.limit("10/day")
async def chat(request: Request, question: Question, background_tasks: BackgroundTasks):
    try:
        response = await chatbot(question.question)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Function to run chatbot and return message
async def chatbot(message):
    response = chat_engine.chat(message)
    return response

# Chat endpoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv('UVICORN_HOST'), port=int(os.getenv('UVICORN_PORT')))