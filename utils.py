# pyrefly: ignore [missing-import]
from PIL import Image, ImageOps
# pyrefly: ignore [missing-import]
import numpy as np
from pathlib import Path

def format(image_source):
    """
    Preprocesses an image for Teachable Machine models using PIL.
    Accepts:
    - String / Path (e.g. "data/ship/image.png")
    - PIL Image object
    - NumPy Array (e.g. from cv2.imread or OpenCV)
    """
    # 1. Convert input to a PIL Image depending on its type
    if isinstance(image_source, (str, Path)):
        # It's a file path
        image = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, np.ndarray):
        # It's a NumPy array (e.g., from OpenCV)
        # OpenCV uses BGR, so swap BGR -> RGB before passing to PIL
        if len(image_source.shape) == 3 and image_source.shape[2] == 3:
            image_source = image_source[:, :, ::-1]  # BGR to RGB
        image = Image.fromarray(image_source).convert("RGB")
    elif isinstance(image_source, Image.Image):
        # It's already a PIL Image
        image = image_source.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_source)}")

    # 2. Resize and crop exactly like Teachable Machine (224x224 with LANCZOS)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # 3. Convert to float32 array and normalize to [-1.0, 1.0]
    image_array = np.asarray(image, dtype=np.float32)
    normalized_image = (image_array / 127.5) - 1.0

    # 4. Add batch dimension: shape (1, 224, 224, 3)
    return np.expand_dims(normalized_image, axis=0)