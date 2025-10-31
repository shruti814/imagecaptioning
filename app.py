import streamlit as st
from caption_segmentation import process_image
from PIL import Image

st.title("VisionaryAI - Image Captioning & Segmentation")
st.write("Upload an image to see segmentation and an AI-generated caption!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Original Image", use_column_width=True)
    with st.spinner("Processing... This might take a few seconds "):
        segmented_image, caption = process_image(uploaded_file)
    st.image(segmented_image, caption="Segmented Image", use_column_width=True)
    st.success(f"**Generated Caption:** {caption}")
