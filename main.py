import streamlit as st
from PIL import Image

import style

st.title("Photo Art Style")

img = st.sidebar.selectbox(
    'Select Image',
    ('amber.jpg', 'cat.jpeg', 'csu.jpg')
)


style_name = st.sidebar.selectbox(
    'Select Style',
    ('candy', 'mosaic', 'rain_princess', 'udnie')
)

model = "saved_models/" + style_name + ".pth"
input_image = "images/content-images/" + img
output_image = "images/output-images" + style_name + "-" + img

st.write("### Source Image:")
image = Image.open(input_image)
st.image(image, width=400)

clicked = st.button("Stylize")

if clicked:
    try:
        model = style.load_model(model)
        style.stylize(model, input_image, output_image)

        st.write("### Output Image:")
        image = Image.open(output_image)
        st.image(image, width=400)
    except Exception as e:
        st.error(f"Error during style transfer: {e}")
