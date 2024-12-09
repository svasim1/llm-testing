import os
import re
import fitz
import pickle
from logging_conf import logger
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage

# Constants
PERSIST_DIR = "storage"
INDEX_FILE = os.path.join(PERSIST_DIR, "index.pkl")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        logger.error(f"Error while extracting text from PDF: {e}")
    return text

# Function to read text from a .txt file
def read_text_from_txt(txt_path):
    text = ""
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read()
    except Exception as e:
        logger.error(f"Error while reading text from TXT file: {e}")
    return text

# Function to split text into chunks
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

# Function to process all files in a directory
def process_files_in_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith(".txt"):
            text = read_text_from_txt(file_path)
        else:
            continue
        chunks = split_text_into_chunks(text)
        documents.extend([Document(text=chunk) for chunk in chunks])
    return documents

# Function to create or load the index
def create_or_load_index():
    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR)
    
    if not os.path.exists(INDEX_FILE):
        documents = process_files_in_directory("../data")
        index = VectorStoreIndex.from_documents(documents)
        with open(INDEX_FILE, "wb") as f:
            pickle.dump(index, f)
        logger.debug(f"Index created and stored in {INDEX_FILE}")
    else:
        with open(INDEX_FILE, "rb") as f:
            index = pickle.load(f)
        logger.debug(f"Index loaded from {INDEX_FILE}")
    
    return index