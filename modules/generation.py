import os
from vertexai.vision_models import MultiModalEmbeddingModel
from vertexai.generative_models import GenerativeModel, Content, Part

from modules.retrieval import find_similar_items

def query_rag_system(question, indexed_items):
    """Query the RAG system with a question."""
    print(f"Processing question: '{question}'")
    
    # Load models
    embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    llm_model = GenerativeModel("gemini-pro-vision")
    
    # Get question embedding
    question_result = embedding_model.get_embeddings(
        image=None,
        contextual_text=question,
        dimension=1408
    )
    question_embedding = question_result.text_embedding
    
    # Find similar items
    top_matches = find_similar_items(question_embedding, indexed_items, top_k=5)
    
    # Prepare context
    text_parts = []
    image_files = []
    
    for match in top_matches:
        if match["type"] == "text":
            text_parts.append(f"[Content from page {match['page']+1}]\n{match['content']}")
        elif match["type"] == "image":
            # Just keep track of the image file paths
            image_files.append({
                "path": match["path"],
                "page": match["page"]+1,
            })
    
    # Combine text context
    text_context = "\n\n".join(text_parts)
    
    # Create prompt
    prompt = f"""
    Answer the following question about the document based on the provided context and images:
    
    QUESTION: {question}
    
    TEXT CONTEXT:
    {text_context}
    
    Provide a comprehensive answer based solely on the information in the context.
    If the information isn't available in the context, please state that clearly.
    """
    
    # Create message content parts
    message_parts = [Part.from_text(prompt)]
    
    # Add images to the same message
    for img_info in image_files:
        try:
            # Load image file
            with open(img_info["path"], "rb") as f:
                image_bytes = f.read()
            
            # Create Part from image bytes
            img_part = Part.from_data(mime_type="image/png", data=image_bytes)
            message_parts.append(img_part)
            
        except Exception as e:
            print(f"Error loading image {img_info['path']}: {e}")
    
    # Create a single content item with user role
    content = [
        Content(
            role="user", 
            parts=message_parts
        )
    ]
    
    # Generate answer
    try:
        response = llm_model.generate_content(content)
        answer = response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        # Fallback to text-only response
        text_only_content = [Content(role="user", parts=[Part.from_text(prompt)])]
        try:
            response = llm_model.generate_content(text_only_content)
            answer = response.text + "\n\n[Note: Images could not be processed due to an error]"
        except Exception as e2:
            answer = f"Error generating response: {e2}\n\nRetrieved context:\n{text_context[:500]}..."
    
    return {
        "question": question,
        "answer": answer,
        "top_matches": top_matches,
        "text_context": text_context
    }

def show_query_result(result):
    """Return a text representation of query results (non-display version)."""
    output = []
    output.append(f"Question: {result['question']}")
    output.append(f"\nAnswer:\n{result['answer']}")

    output.append("\nTop Matching Items:")
    for i, match in enumerate(result["top_matches"]):
        output.append(f"\n--- Match {i+1} (similarity: {match['similarity']:.4f}) ---")
        output.append(f"Type: {match['type']}")
        output.append(f"Page: {match['page']+1}")

        if match["type"] == "text":
            # Show truncated text for readability
            if len(match["content"]) > 300:
                output.append(f"Content (truncated): {match['content'][:300]}...")
            else:
                output.append(f"Content: {match['content']}")
        else:
            output.append(f"Image path: {match['path']}")
            
    return "\n".join(output)