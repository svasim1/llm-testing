import os
from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from slowapi import Limiter
from slowapi.util import get_remote_address
from dotenv import load_dotenv
from database import get_db, authenticate_user, get_current_user, create_access_token, TokenData
from chat import chatbot, token_usage_stats
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from models import User
from datetime import timedelta

# Load the environment variables - see .env.sample
load_dotenv()

# Constants
ACCESS_TOKEN_EXPIRE_MINUTES = 30
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS")

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Setup FastAPI
app = FastAPI(
    title="Chatbot API",
    description="A simple chatbot API for riksdagstracker.se",
    version="1.0.0",
    swagger_ui_parameters={
        "deepLinking": True,
        "displayOperationId": True,
        "docExpansion": "none",
        "filter": True,
        "showExtensions": True,
    }
)
app.state.limiter = limiter

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str

class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, pattern=r'^[\s\S]*$')

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
    response_context, sources = await chatbot(question.question, current_user.username)
    return {"response": response_context, "sources": sources}

# Endpoint to get total token usage statistics (for current session)
@app.get("/token-usage")
async def get_token_usage():
    return token_usage_stats

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("UVICORN_HOST"), port=int(os.getenv("UVICORN_PORT")))