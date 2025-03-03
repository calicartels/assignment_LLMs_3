import os
import os
import streamlit as st

# Project settings
PROJECT_ID = "capstone-449418"
LOCATION = "us-central1"

# Base directory is relative to the application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Storage directories
DATA_DIR = os.path.join(BASE_DIR, "data")
CREDENTIALS_DIR = os.path.join(BASE_DIR, "credentials")
OUTPUT_DIR = os.path.join(BASE_DIR, "static")
INDEX_DIR = os.path.join(OUTPUT_DIR, "index")
TEXT_DIR = os.path.join(OUTPUT_DIR, "text")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")

# File paths - uses environment variables with fallbacks to the directories
KEY_PATH = None
if not hasattr(st, 'secrets') or 'google_credentials' not in st.secrets:
    # Only use local file if secrets not available
    KEY_PATH = os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS", 
        os.path.join(CREDENTIALS_DIR, "capstone-449418-38bd0569f608.json")
    )

# PDF path will fall back to a default sample.pdf in the data directory
PDF_PATH = os.environ.get(
    "PDF_FILE_PATH",
    os.path.join(DATA_DIR, "2403.03206v1.pdf")
)

# Ensure all necessary directories exist
for dir_path in [DATA_DIR, CREDENTIALS_DIR, INDEX_DIR, TEXT_DIR, IMAGE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Default index file
DEFAULT_INDEX_PATH = os.path.join(INDEX_DIR, "rag_index.json")