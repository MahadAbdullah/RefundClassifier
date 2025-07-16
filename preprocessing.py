import numpy as np
from PIL import Image
import io

def preprocess_image_bytes(image_bytes, target_size=(224, 224)):
    """
    Preprocesses an image given as raw bytes.
    - Resizes to target_size
    - Converts to float32
    - Normalizes to [0,1]
    - Adds batch dimension
    Returns: image array with shape (1, 224, 224, 3)
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
