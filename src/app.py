

import streamlit as st
import cv2
from PIL import Image
import numpy as np
from model_inference import load_model, detect_objects

st.title('Object Detection with YOLO')
model = load_model('/content/')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button('Detect Objects'):
        confidence = st.slider('Confidence threshold', 0.0, 1.0, 0.5, 0.01)
        threshold = st.slider('NMS threshold', 0.0, 1.0, 0.3, 0.01)
        output_image = detect_objects(model, image, confidence, threshold)
        output_image = Image.fromarray(output_image.astype('uint8'), 'RGB')
        st.image(output_image, caption='Detected Objects', use_column_width=True)
