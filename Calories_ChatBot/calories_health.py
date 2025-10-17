import streamlit as st
import os
import pathlib
import textwrap
from PIL import Image
import google.generativeai as genai


# Load API key securely: prefer environment variable, fall back to Streamlit secrets if present.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    try:
        # Access Streamlit secrets only when configured; Streamlit raises if none exist.
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        GEMINI_API_KEY = None

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Set the GEMINI_API_KEY environment variable or configure Streamlit secrets securely (do not commit secrets to your repo).")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

def get_gemini_response(user_input, image, prompt):
    model = genai.GenerativeModel('gemini-2.5-flash')
    try:
        response = model.generate_content([user_input, image[0], prompt])
        return response.text
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return None

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type" : uploaded_file.type, # Get the mime type of the uploaded file
                "data" : bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")


st.set_page_config(page_title="Gemini Image Demo")

st.header("Calories & Health AI Assistant 🍎")
user_input = st.text_input("Ask me something about calories, fitness, or health: ", key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
image=""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_container_width=True)

submit = st.button("Tell me about the image")

input_prompt = """ 
               You are an expert in nutritionist where you need to see the food items from the image
               and calculate the total calories, also provide the details of every food items with calories intake
               is below format

               1. Item 1 - no of calories
               2. Item 2 - no of calories
               ----
               ----
"""
## If ask button is clicked

if submit:
    if uploaded_file is None:
        st.error("Please upload an image before submitting.")
    else:
        image_data = input_image_setup(uploaded_file)
        response = get_gemini_response(user_input, image_data, input_prompt)
        if response:
            st.subheader("The Response is")
            st.write(response)

