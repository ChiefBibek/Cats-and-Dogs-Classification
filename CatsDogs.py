import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.models import load_model

model = load_model('model.keras')

#preprocessing the image
def preprocess(image):
    image = image.resize((64, 64))  # Resize the image
    # Convert image to numpy array and normalize if required
    # image = np.array(image)  # Normalize to [0, 1] if required
    image = np.reshape(image, [1, 64, 64, 3])  # Reshape to match model input
    return image



st.title('Cats and Dogs Image Classification')
# File uploader for user-uploaded image
selected_img = st.file_uploader('Upload an image of Dogs or Cats:', type=['png', 'jpg', 'jpeg'])

if selected_img is not None:
    # Display the uploaded image
    image = Image.open(selected_img)
    st.image(selected_img, width=300)
    image = preprocess(image)
else:
    st.warning('Please upload an image.')
    
#button for recognizing the selected image
if st.button('Recognize the selected image'):
    if image is not None:
        # Prediction
        img_pred = model.predict(image)
        
        # Display the recognized image
        if img_pred == 0:
            st.success('The image is a Dog.')
        else:
            st.success('The image is a Cat.')