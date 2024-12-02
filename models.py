from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from passlib.context import CryptContext
import os

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def create_user(db, username, email, password):
    hashed_password = get_password_hash(password)
    db_user = User(username=username, email=email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# Create the database tables
Base.metadata.create_all(bind=engine)

def get_user(db, user_id):
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_username(db, username):
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db, email):
    return db.query(User).filter(User.email == email).first()

def get_users(db, skip=0, limit=10):
    return db.query(User).offset(skip).limit(limit).all()