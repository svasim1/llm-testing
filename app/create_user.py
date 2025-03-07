import os
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from models import Base, create_user, clear_database, get_password_hash, User
from models import SessionLocal

# Load environment variables from .env file
load_dotenv()

# Create a new database session
db: Session = SessionLocal()

# Create the users table if it doesn't exist
Base.metadata.create_all(bind=db.get_bind())

# Clear the database
clear_database(db)

# Create a specific user with specified details
def create_specific_user(db: Session, user_id: int, username: str, email: str, password: str):
    hashed_password = get_password_hash(password)
    db_user = User(id=user_id, username=username, email=email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# Create the specific user
specific_user = create_specific_user(db, 1, 'admin', 'test@example.com', 'password1')
print(f"Specific user created successfully: {specific_user.username}")

# Close the database session
db.close()