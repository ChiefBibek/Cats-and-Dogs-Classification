import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

model = load_model('model.keras')

st.title('Cats and Dogs Image Classification')
# File uploader for user-uploaded image
selected_img = st.file_uploader('Upload an image of Dogs or Cats:', type=['png', 'jpg', 'jpeg'])

if selected_img is not None:
    # Display the uploaded image
    st.image(selected_img, width=300)
    image = plt.imread(selected_img)

    
