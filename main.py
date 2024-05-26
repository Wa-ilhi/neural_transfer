import streamlit as st
from PIL import Image
import style
import os

# Page title
st.title("Photo Art Style Transfer")

# Sidebar
st.sidebar.title("Settings")

# Upload an image
uploaded_file = st.sidebar.file_uploader(
    "Upload your image", type=["jpg", "jpeg", "png"])

# Select style
style_name = st.sidebar.selectbox(
    "Select Style", ('candy', 'mosaic', 'rain-princess', 'udnie'))

# Slider for adjusting quality
num_steps = st.sidebar.slider(
    'Adjust Quality (num_steps)', min_value=100, max_value=1000, value=500, step=100)

# Stylize button
clicked = st.sidebar.button("Stylize")

# Main content area
st.write("### Stylized Image")

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Source Image", use_column_width=True)

    if clicked:
        try:
            # Stylize the image
            model_path = os.path.join(
                "saved_models", style_name.lower() + ".pth")
            output_image_path = style_name.lower() + "-output.png"

            model = style.load_model(model_path)
            style.stylize(model, uploaded_file,
                          output_image_path, num_steps=num_steps)

            output_image = Image.open(output_image_path)
            st.image(output_image, caption="Stylized Image",
                     use_column_width=True)

            # Share on Social Media
            st.write("### Share on Social Media")
            facebook_url = "https://www.facebook.com/sharer/sharer.php?u=" + output_image_path
            instagram_url = "https://www.instagram.com/"

            st.markdown(
                '<a href="' + facebook_url + '" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" width="50" style="margin-right: 20px;"></a>'

                '<a href="' + instagram_url +
                '" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png" width="50"></a>',
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error during style transfer: {e}")
else:
    st.write("Please upload an image to apply the style.")
