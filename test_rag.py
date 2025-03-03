"""
Test script for the Multimodal RAG system.
Run this to verify everything works before deployment.
"""
import os
import sys
from utils.auth import setup_google_auth
from modules.extraction import extract_from_pdf
from modules.embedding import create_embeddings
from modules.retrieval import save_index, load_index
from modules.generation import query_rag_system, show_query_result
import config

def test_complete_pipeline(pdf_path, question):
    """Test the complete pipeline from PDF to answer."""
    print("=== Testing Complete Pipeline ===")
    print(f"PDF: {pdf_path}")
    print(f"Question: {question}")
    print("--------------------------------")
    
    # Setup authentication
    setup_google_auth(config.KEY_PATH)
    
    # Extract content from PDF
    print("Step 1: Extracting content from PDF...")
    extracted_items = extract_from_pdf(pdf_path)
    print(f"Extracted {len(extracted_items)} items.")
    
    # Generate embeddings
    print("\nStep 2: Generating embeddings...")
    indexed_items = create_embeddings(extracted_items)
    print(f"Generated embeddings for {len(indexed_items)} items.")
    
    # Save index
    print("\nStep 3: Saving index...")
    save_index(indexed_items)
    
    # Load index (to verify it saved correctly)
    print("\nStep 4: Loading index...")
    loaded_items = load_index()
    print(f"Loaded {len(loaded_items)} items from index.")
    
    # Query the system
    print("\nStep 5: Querying the system...")
    result = query_rag_system(question, loaded_items)
    
    # Show results
    print("\nStep 6: Results:")
    print("--------------")
    print(show_query_result(result))
    
    return result

def test_load_and_query(question):
    """Test loading an existing index and querying it."""
    print("=== Testing Load and Query ===")
    print(f"Question: {question}")
    print("----------------------------")
    
    # Setup authentication
    setup_google_auth(config.KEY_PATH)
    
    # Load index
    print("Step 1: Loading index...")
    indexed_items = load_index()
    print(f"Loaded {len(indexed_items)} items from index.")
    
    # Query the system
    print("\nStep 2: Querying the system...")
    result = query_rag_system(question, indexed_items)
    
    # Show results
    print("\nStep 3: Results:")
    print("--------------")
    print(show_query_result(result))
    
    return result

# Modify this part in test_rag.py
if __name__ == "__main__":
    import json
    
    # Delete any existing empty index
    if os.path.exists(config.DEFAULT_INDEX_PATH):
        try:
            with open(config.DEFAULT_INDEX_PATH, 'r') as f:
                index_data = json.load(f)
                if len(index_data) == 0:
                    print("Found empty index file. Deleting it to reprocess PDF...")
                    os.remove(config.DEFAULT_INDEX_PATH)
        except (json.JSONDecodeError, FileNotFoundError):
            print("Found invalid index file. Deleting it to reprocess PDF...")
            os.remove(config.DEFAULT_INDEX_PATH)
    
    # Check if index exists
    if os.path.exists(config.DEFAULT_INDEX_PATH):
        # If it does, just test querying
        test_load_and_query("What is flow matching ?")
    else:
        # Otherwise, test the complete pipeline
        print(f"Running complete pipeline with PDF: {config.PDF_PATH}")
        
        # Verify PDF exists
        if not os.path.exists(config.PDF_PATH):
            print(f"Error: PDF file not found at {config.PDF_PATH}")
            print(f"Please place your PDF file at {config.PDF_PATH}")
            print("Or run with: PDF_FILE_PATH=/path/to/pdf python test_rag.py")
            exit(1)
        
        test_complete_pipeline(config.PDF_PATH, "What are the key improvements in Stable Diffusion 3?")