import os
import re
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import Depends, status, FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from openai import OpenAI
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Document
from sqlalchemy.orm import Session
from models import SessionLocal, get_user, get_user_by_username, User

# Setup logging
log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_directory, log_filename), encoding="utf-8"),
        #logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load the environment variables - see .env.sample
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Constants
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = 30
PERSIST_DIR = os.getenv("PERSIST_DIR")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS")

# OAuth2 and password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class UserInDB(User):
    __allow_unmapped__ = True
    hashed_password: str

class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, pattern=r'^[a-zA-Z0-9\s\.,?!åäöÅÄÖ":;-]+$')

# Functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(db: Session, username: str, password: str):
    user = get_user_by_username(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user_by_username(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def moderate_content(content: str):
    response = client.moderations.create(input=content)
    if response.results[0].flagged:
        raise HTTPException(status_code=400, detail="Content flagged as unsafe")

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
with open("data/lagbok.txt", "r", encoding="utf-8") as file:
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
        print(f"Index stored in {PERSIST_DIR}")
    else:
        # Load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        print(f"Index loaded from {PERSIST_DIR}")
except Exception as e:
    print(f"Error while setting up the index: {e}")
    raise

# Create retriever from index
retriever = index.as_retriever()

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

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Setup FastAPI
app = FastAPI()
app.state.limiter = limiter

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Routes
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/chat")
@limiter.limit("10/day")
async def chat(request: Request, question: Question, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    logger.debug(f"Received question: {question.question}")
    logger.debug(f"Current user: {current_user.username}")
    try:
        response_context, sources = await chatbot(question.question, current_user.username)
        logger.debug(f"Chatbot response: {response_context}")
        logger.debug(f"Sources: {sources}")
        return {"response": response_context, "sources": sources}
    except HTTPException as e:
        logger.error(f"HTTPException in chat endpoint: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Get user by ID
async def chatbot(message, user_id):
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
    
    # Log token usage details
    logger.debug(f"Prompt tokens: {response.usage.prompt_tokens}")
    logger.debug(f"Completion tokens: {response.usage.completion_tokens}")
    logger.debug(f"Total tokens: {response.usage.total_tokens}")
    
    return response_content, sources

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("UVICORN_HOST"), port=int(os.getenv("UVICORN_PORT")))