import io
import requests
# pyrefly: ignore [missing-import]
from PIL import Image, ImageOps
# pyrefly: ignore [missing-import]
import numpy as np

def format(image_source):
    """
    Preprocesses an image for Teachable Machine models using PIL.
    Accepts:
    - Web URL string ("https://...")
    - Local file path string ("data/ship/1.png")
    - Raw bytes (e.g. uploaded file from FastAPI)
    - PIL Image object
    - NumPy Array (e.g. from OpenCV)
    """
    # 1. Convert input source into a PIL Image
    if isinstance(image_source, str):
        if image_source.startswith("http://") or image_source.startswith("https://"):
            # Fetch image from URL
            response = requests.get(image_source, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            # Local file path
            image = Image.open(image_source).convert("RGB")
            
    elif isinstance(image_source, bytes):
        # Raw bytes stream
        image = Image.open(io.BytesIO(image_source)).convert("RGB")
        
    elif isinstance(image_source, np.ndarray):
        # NumPy array (e.g., OpenCV BGR to RGB)
        if len(image_source.shape) == 3 and image_source.shape[2] == 3:
            image_source = image_source[:, :, ::-1]
        image = Image.fromarray(image_source).convert("RGB")
        
    elif isinstance(image_source, Image.Image):
        # Already a PIL Image
        image = image_source.convert("RGB")
        
    else:
        raise TypeError(f"Unsupported image input type: {type(image_source)}")

    # 2. Resize and crop to 224x224 (Teachable Machine pipeline)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # 3. Convert to float32 array and normalize to [-1.0, 1.0]
    image_array = np.asarray(image, dtype=np.float32)
    normalized_image = (image_array / 127.5) - 1.0

    # 4. Add batch dimension -> shape (1, 224, 224, 3)
    return np.expand_dims(normalized_image, axis=0)