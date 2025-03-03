# General helper functions
import os
import json
import base64
from PIL import Image as PILImage

def ensure_directory(dir_path):
    """Make sure a directory exists."""
    os.makedirs(dir_path, exist_ok=True)
    
def save_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f)
        
def load_json(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)
        
def image_to_base64(image_path):
    """Convert an image to base64 encoding."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
        
def base64_to_image(base64_data, output_path):
    """Convert base64 data to an image file."""
    image_data = base64.b64decode(base64_data)
    with open(output_path, "wb") as f:
        f.write(image_data)