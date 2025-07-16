import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# Load the ResNet50 model pre-trained on ImageNet
model = ResNet50(weights='imagenet')

def predict(preprocessed_image):
    """
    Takes a preprocessed image (normalized to [0,1] and shaped (1, 224, 224, 3)),
    rescales it to [0,255], applies ResNet50 preprocessing, and returns
    top-3 predictions in dictionary form: {label: probability}.
    """
    image_for_model = preprocess_input(preprocessed_image * 255.0)
    preds = model.predict(image_for_model)
    decoded = decode_predictions(preds, top=3)[0]
    return {label: float(prob) for (_, label, prob) in decoded}
