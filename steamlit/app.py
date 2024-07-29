import streamlit as st
import tensorflow as tf
import model
import numpy as np
from PIL import Image

st.title("brain scan")

img=st.file_uploader("upload scan file")

if img:
    st.image(img)

    Model = model.main()
    Model.load_weights("model_weights.h5")
    img = Image.open(img)
    img = img.resize((256,256))
    img = np.array(img) / 255.0  
    img = np.expand_dims(img, axis=0)

    predictions = Model.predict(img)
    classes = ["glioma_tumor","meningioma_tumor","normal","pituitary_tumor"]
    predicted_class = np.argmax(predictions[0])
    st.write("Predicted class:", classes[predicted_class])

