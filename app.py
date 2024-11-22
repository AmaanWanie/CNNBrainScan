import streamlit as st
import tensorflow as tf
import model
import numpy as np
from PIL import Image


st.title("Brain Scan Classification")


uploaded_file = st.file_uploader("Upload a brain scan file (e.g., JPG, PNG)")


if uploaded_file is not None:
   
    st.image(uploaded_file, caption="Uploaded Scan", use_column_width=True)
    
  
    brain_scan_model = model.main()
    brain_scan_model.load_weights("model_weights.h5")
    
   
    image = Image.open(uploaded_file)
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    

    predictions = brain_scan_model.predict(image_array)
    class_labels = ["Glioma Tumor", "Meningioma Tumor", "Normal", "Pituitary Tumor"]
    predicted_label = np.argmax(predictions[0])
    
   
    st.write("Predicted Class:", class_labels[predicted_label])
