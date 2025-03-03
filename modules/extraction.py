import os
import pymupdf
import base64
from tqdm import tqdm

from config import TEXT_DIR, IMAGE_DIR

def extract_from_pdf(pdf_path):
    """Extract text and images from PDF."""
    print(f"Processing PDF: {pdf_path}")
    doc = pymupdf.open(pdf_path)

    items = []

    # Process each page
    for page_num, page in enumerate(tqdm(doc, desc="Processing pages")):
        # Extract text
        text = page.get_text()

        # Chunk the text
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 10:  # Skip empty chunks
                item_id = f"text_{page_num}_{i}"
                file_path = os.path.join(TEXT_DIR, f"{item_id}.txt")

                # Save to file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(chunk)

                items.append({
                    "id": item_id,
                    "type": "text",
                    "content": chunk,
                    "page": page_num,
                    "path": file_path
                })

        # Extract images
        images = page.get_images(full=True)

        for i, img_info in enumerate(images):
            xref = img_info[0]

            try:
                # Extract image
                base_img = doc.extract_image(xref)

                if base_img:
                    item_id = f"image_{page_num}_{i}"
                    file_path = os.path.join(IMAGE_DIR, f"{item_id}.png")

                    # Save image
                    with open(file_path, "wb") as f:
                        f.write(base_img["image"])

                    # Store base64 for API calls
                    img_base64 = base64.b64encode(base_img["image"]).decode("utf-8")

                    items.append({
                        "id": item_id,
                        "type": "image",
                        "content": img_base64,
                        "page": page_num,
                        "path": file_path
                    })
            except Exception as e:
                print(f"Error extracting image {xref} on page {page_num}: {e}")

    print(f"Extracted {len(items)} items ({len([i for i in items if i['type']=='text'])} text chunks and {len([i for i in items if i['type']=='image'])} images)")
    return items

def chunk_text(text, chunk_size=800, overlap=100):
    """Split text into chunks with overlap, ensuring no chunk exceeds 1024 characters."""
    # First split by paragraphs
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If this paragraph alone is too big, split it by sentences
        if len(paragraph) > chunk_size:
            # Split into sentences (simple split by period)
            sentences = paragraph.replace('. ', '.|').split('|')
            for sentence in sentences:
                if len(sentence) > chunk_size:
                    # If even a sentence is too big, split by a fixed length
                    for i in range(0, len(sentence), chunk_size // 2):
                        sub_chunk = sentence[i:i + chunk_size // 2]
                        if sub_chunk:
                            chunks.append(sub_chunk)
                elif len(current_chunk) + len(sentence) + 2 > chunk_size:
                    # Save current chunk and start a new one
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
        # If adding this paragraph exceeds chunk size, save current chunk and start a new one
        elif len(current_chunk) + len(paragraph) + 2 > chunk_size and current_chunk:
            chunks.append(current_chunk)
            
            # Keep some overlap from the previous chunk
            words = current_chunk.split()
            overlap_words = min(len(words), overlap//10)  # Approximately 10 chars per word
            overlap_text = " ".join(words[-overlap_words:]) if overlap_words > 0 else ""
            current_chunk = overlap_text + " " + paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    # Final safety check - ensure all chunks are under 1000 chars
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= 1000:
            final_chunks.append(chunk)
        else:
            # Split into smaller chunks
            for i in range(0, len(chunk), 900):  # Use 900 to be safe
                sub_chunk = chunk[i:i + 900]
                if sub_chunk:
                    final_chunks.append(sub_chunk)
    
    return final_chunks

def verify_chunk_sizes(items):
    """Check text chunks to ensure they're under 1024 characters."""
    too_large = []
    for item in items:
        if item['type'] == 'text':
            length = len(item['content'])
            if length > 1000:  # Checking against a slightly lower threshold to be safe
                too_large.append((item['id'], length))
    
    if too_large:
        print(f"Found {len(too_large)} chunks that are too large:")
        for item_id, length in too_large:
            print(f"  - {item_id}: {length} characters")
    else:
        print("All chunks are within the size limit!")
    
    return too_large