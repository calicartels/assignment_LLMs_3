"""
Multimodal RAG Application
--------------------------
Main entry point for the multimodal retrieval-augmented generation system.
"""
import os
import argparse
from utils.auth import setup_google_auth
from modules.extraction import extract_from_pdf
from modules.embedding import create_embeddings
from modules.retrieval import save_index, load_index
from modules.generation import query_rag_system, show_query_result
import config

def process_pdf(pdf_path):
    """Process a PDF document and create embeddings."""
    # Extract content from PDF
    print("Extracting content from the paper...")
    extracted_items = extract_from_pdf(pdf_path)

    # Generate embeddings
    print("Generating embeddings...")
    indexed_items = create_embeddings(extracted_items)

    # Save the index
    save_index(indexed_items)
    
    return indexed_items

def process_query(question, indexed_items=None):
    """Process a question against the indexed items."""
    # Load index if not provided
    if indexed_items is None:
        if not os.path.exists(os.path.join(config.INDEX_DIR, "rag_index.json")):
            raise ValueError("No index found. Please process a PDF first.")
        indexed_items = load_index()
        
    # Query the system
    result = query_rag_system(question, indexed_items)
    
    # Show results
    print(show_query_result(result))
    
    return result

def main():
    """Main function to parse arguments and run the application."""
    parser = argparse.ArgumentParser(description="Multimodal RAG System")
    parser.add_argument("--pdf", type=str, help="Path to PDF file to process")
    parser.add_argument("--query", type=str, help="Question to ask about the document")
    parser.add_argument("--key", type=str, help="Path to Google Cloud credentials JSON file")
    
    args = parser.parse_args()
    
    # Set up Google Cloud authentication
    if args.key:
        setup_google_auth(args.key)
    else:
        setup_google_auth(config.KEY_PATH)
    
    # Process PDF if provided
    if args.pdf:
        indexed_items = process_pdf(args.pdf)
        
        # If query is also provided, process it
        if args.query:
            process_query(args.query, indexed_items)
    # Otherwise, just process the query if provided
    elif args.query:
        process_query(args.query)
    else:
        print("Please provide either a PDF file to process (--pdf) or a question to ask (--query).")
        
if __name__ == "__main__":
    main()