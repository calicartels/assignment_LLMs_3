import numpy as np
from tqdm import tqdm
from vertexai.vision_models import Image as VertexImage
from vertexai.vision_models import MultiModalEmbeddingModel

def create_embeddings(items):
    """Generate embeddings for all items using Google's multimodal embedding model."""
    print("Generating embeddings...")

    # Load the embedding model
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    embedding_dim = 1408  # Standard dimension for this model

    # Process each item
    for item in tqdm(items, desc="Embedding items"):
        try:
            if item["type"] == "text":
                # Generate text embedding
                result = model.get_embeddings(
                    image=None,
                    contextual_text=item["content"],
                    dimension=embedding_dim
                )
                item["embedding"] = np.array(result.text_embedding)

            elif item["type"] == "image":
                # Generate image embedding
                img = VertexImage.load_from_file(item["path"])

                # Use page context for better relevance
                context = f"Image from page {item['page']+1} of the document"

                result = model.get_embeddings(
                    image=img,
                    contextual_text=context,
                    dimension=embedding_dim
                )
                item["embedding"] = np.array(result.image_embedding)

        except Exception as e:
            print(f"Error generating embedding for {item['id']}: {e}")
            item["embedding"] = None

    # Keep only items with valid embeddings
    valid_items = [item for item in items if item.get("embedding") is not None]
    print(f"Successfully embedded {len(valid_items)} out of {len(items)} items")

    return valid_items