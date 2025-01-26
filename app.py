import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('fundus_disease_model.h5')

# Define label names and their full forms
labels = {
    'N': 'No abnormalities detected',
    'D': 'Diabetic Retinopathy',
    'G': 'Glaucoma',
    'C': 'Cataract',
    'A': 'Age-related Macular Degeneration',
    'H': 'Hypertension',
    'M': 'Myopia',
    'O': 'Other Diseases'
}

# Define a function to preprocess the uploaded images
def preprocess_image(image, img_size=(224, 224)):
    image = image.resize(img_size).convert('RGB')
    image = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Streamlit app layout
st.title("Fundus Disease Detection App")

st.write("Upload Left and Right Fundus Images to Detect Possible Diseases")

left_image = st.file_uploader("Upload Left Fundus Image", type=["jpg", "png", "jpeg"], key="left")
right_image = st.file_uploader("Upload Right Fundus Image", type=["jpg", "png", "jpeg"], key="right")

if left_image and right_image:
    # Load and preprocess images
    left_img = Image.open(left_image)
    right_img = Image.open(right_image)

    left_input = preprocess_image(left_img)
    right_input = preprocess_image(right_img)

    # Make predictions
    left_pred = model.predict(left_input)
    right_pred = model.predict(right_input)

    # Interpret predictions
    left_diseases = [labels[key] for i, (key, val) in enumerate(labels.items()) if left_pred[0][i] > 0.5]
    right_diseases = [labels[key] for i, (key, val) in enumerate(labels.items()) if right_pred[0][i] > 0.5]

    # Display results
    st.subheader("Prediction Results")

    st.write("### Left Fundus:")
    if left_diseases and 'No abnormalities detected' not in left_diseases:
        st.markdown(f"<span style='color:red; font-weight:bold;'>Disease Detected: {', '.join(left_diseases)}!</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:green; font-weight:bold;'>No abnormalities detected</span>", unsafe_allow_html=True)

    st.write("### Right Fundus:")
    if right_diseases and 'No abnormalities detected' not in right_diseases:
        st.markdown(f"<span style='color:red; font-weight:bold;'>Disease Detected: {', '.join(right_diseases)}!</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:green; font-weight:bold;'>No abnormalities detected</span>", unsafe_allow_html=True)

    # Show images
    st.image(left_img, caption="Left Fundus Image", use_container_width=True)
    st.image(right_img, caption="Right Fundus Image", use_container_width=True)
else:
    st.info("Please upload both Left and Right Fundus images.")
