import os
import json
import numpy as np
from PIL import Image as PILImage
import base64

from config import INDEX_DIR

def find_similar_items(query_embedding, items, top_k=5):
    """Find most similar items using cosine similarity."""

    # Calculate cosine similarity manually
    def cosine_similarity(v1, v2):
        # Dot product
        dot_product = np.dot(v1, v2)
        # Magnitudes
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        # Avoid division by zero
        if norm_v1 == 0 or norm_v2 == 0:
            return 0
        # Cosine similarity
        return dot_product / (norm_v1 * norm_v2)

    # Calculate similarity for each item
    similarities = []
    for idx, item in enumerate(items):
        if "embedding" in item and item["embedding"] is not None:
            sim = cosine_similarity(query_embedding, item["embedding"])
            similarities.append((idx, sim))

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top k matches
    results = []
    for idx, sim in similarities[:top_k]:
        item_copy = items[idx].copy()
        item_copy["similarity"] = float(sim)
        results.append(item_copy)

    return results

def save_index(items, filename=None):
    """Save indexed items to disk."""
    if filename is None:
        filename = os.path.join(INDEX_DIR, "rag_index.json")
        
    # Create a copy without large data
    save_items = []

    for item in items:
        save_item = item.copy()

        # Convert numpy array to list for JSON serialization
        if "embedding" in save_item:
            save_item["embedding"] = save_item["embedding"].tolist()

        # Replace base64 with placeholder to save space
        if save_item["type"] == "image" and "content" in save_item:
            save_item["content"] = "[BASE64_IMAGE]"

        save_items.append(save_item)

    # Save to JSON
    with open(filename, "w") as f:
        json.dump(save_items, f)

    print(f"Saved index with {len(save_items)} items to {filename}")

def load_index(filename=None):
    """Load indexed items from disk."""
    if filename is None:
        filename = os.path.join(INDEX_DIR, "rag_index.json")
        
    with open(filename, "r") as f:
        items = json.load(f)

    # Convert lists back to numpy arrays
    for item in items:
        if "embedding" in item:
            item["embedding"] = np.array(item["embedding"])

        # Reload images if needed
        if item["type"] == "image" and item["content"] == "[BASE64_IMAGE]":
            try:
                with open(item["path"], "rb") as f:
                    item["content"] = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                print(f"Error reloading image {item['path']}: {e}")

    print(f"Loaded index with {len(items)} items")
    return items

def show_item(item):
    """Return a text representation of an item (non-display version)."""
    output = []
    output.append(f"ID: {item['id']} | Type: {item['type']} | Page: {item['page']+1}")

    if item["type"] == "text":
        output.append("\nContent:")
        output.append(item["content"])
    elif item["type"] == "image":
        output.append(f"\nImage path: {item['path']}")
        
    return "\n".join(output)