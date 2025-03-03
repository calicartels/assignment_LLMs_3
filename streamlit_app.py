import streamlit as st
import os
import time
import json
from PIL import Image

from utils.auth import setup_google_auth
from modules.extraction import extract_from_pdf
from modules.embedding import create_embeddings
from modules.retrieval import save_index, load_index
from modules.generation import query_rag_system
import config

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="📚",
    layout="wide"
)

# Initialize Google Cloud authentication
setup_google_auth(config.KEY_PATH)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        background-color: #4a6fa5;
        color: white;
        font-weight: bold;
    }
    .stTextArea textarea {
        font-size: 1rem;
    }
    .evidence-header {
        font-weight: bold;
        color: #166d97;
        margin-bottom: 10px;
    }
    .evidence-item {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 4px solid #47b8e0;
    }
    .image-evidence {
        text-align: center;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Application title
    st.title("Multimodal Document RAG System")
    
    # Initialize session state if needed
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'index_path' not in st.session_state:
        st.session_state.index_path = None
    if 'has_index' not in st.session_state:
        st.session_state.has_index = os.path.exists(config.DEFAULT_INDEX_PATH)
    if 'processed_items' not in st.session_state:
        st.session_state.processed_items = 0
    
    # Create tabs for Upload and Query
    tab1, tab2 = st.tabs(["Upload Document", "Ask Questions"])
    
    # Upload Document Tab
    with tab1:
        st.header("Upload a PDF Document")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
        
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            with open(os.path.join(config.DATA_DIR, "temp.pdf"), "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Process button
            if st.button("Process Document"):
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Extract content
                    status_text.text("Extracting content from PDF...")
                    progress_bar.progress(10)
                    
                    pdf_path = os.path.join(config.DATA_DIR, "temp.pdf")
                    extracted_items = extract_from_pdf(pdf_path)
                    
                    progress_bar.progress(40)
                    status_text.text(f"Generated {len(extracted_items)} items. Creating embeddings...")
                    
                    # Generate embeddings
                    indexed_items = create_embeddings(extracted_items)
                    
                    progress_bar.progress(80)
                    status_text.text("Saving index...")
                    
                    # Create a unique ID for this session
                    session_id = int(time.time())
                    st.session_state.session_id = session_id
                    
                    # Save with session ID in filename
                    index_path = os.path.join(config.INDEX_DIR, f"index_{session_id}.json")
                    save_index(indexed_items, filename=index_path)
                    st.session_state.index_path = index_path
                    st.session_state.has_index = True
                    st.session_state.processed_items = len(indexed_items)
                    
                    progress_bar.progress(100)
                    status_text.text("")
                    
                    # Success message with item counts
                    text_count = len([i for i in indexed_items if i['type'] == 'text'])
                    image_count = len([i for i in indexed_items if i['type'] == 'image'])
                    
                    st.success(f"""
                    📄 Document processed successfully!
                    - Total items: {len(indexed_items)}
                    - Text chunks: {text_count}
                    - Images: {image_count}
                    """)
                    
                    # Add guidance to switch to the query tab
                    st.info("👉 Switch to the 'Ask Questions' tab to start querying your document.")
                    
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
    
    # Ask Questions Tab
    with tab2:
        st.header("Ask Questions About Your Document")
        
        if not st.session_state.has_index:
            st.warning("⚠️ Please upload and process a document first (in the Upload Document tab).")
        else:
            # Input for the question
            question = st.text_area("Enter your question:", height=100)
            
            # Submit button for the question
            if st.button("Submit Question"):
                if question.strip():
                    # Show a spinner while processing
                    with st.spinner("Processing your question..."):
                        try:
                            # Determine which index to use
                            if st.session_state.index_path:
                                index_path = st.session_state.index_path
                            else:
                                index_path = config.DEFAULT_INDEX_PATH
                            
                            # Load index and process query
                            indexed_items = load_index(index_path)
                            result = query_rag_system(question, indexed_items)
                            
                            # Display results
                            st.subheader("Answer")
                            st.write(result["answer"])
                            
                            # Show supporting evidence
                            st.markdown("---")
                            st.subheader("Supporting Evidence")
                            
                            # Create columns for the evidence items
                            for i, match in enumerate(result["top_matches"]):
                                with st.expander(f"Evidence {i+1}: {match['type'].upper()} (Page {match['page']+1}, Similarity: {match['similarity']:.2f})"):
                                    if match["type"] == "text":
                                        st.markdown(f"{match['content']}")
                                    else:  # Image type
                                        try:
                                            img = Image.open(match["path"])
                                            st.image(img, caption=f"Image from page {match['page']+1}", use_column_width=True)
                                        except Exception as e:
                                            st.error(f"Error loading image: {str(e)}")
                        
                        except Exception as e:
                            st.error(f"Error processing question: {str(e)}")
                else:
                    st.warning("Please enter a question.")

if __name__ == "__main__":
    main()