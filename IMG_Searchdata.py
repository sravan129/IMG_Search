import os
import base64
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# Utility function: Encode uploaded image to Base64
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode()


# Streamlit UI
st.set_page_config(page_title="Image Q&A with GPT-4o", page_icon="📸")
st.title("Image Understanding App")
st.header("API Settings")
OPENAI_API_KEY = st.text_input("Enter your OpenAI API Key", type="password")

# Stop if no API key entered
if not OPENAI_API_KEY:
    st.warning("Please enter your OpenAI API Key to continue.")
    st.stop()

# Initialize model
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0)

# Define the multimodal prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that can describe and analyze images."),
    (
        "human",
        [
            {"type": "text", "text": "{input}"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,{image}", "detail": "high"},
            },
        ],
    ),
])

# Combine the prompt with the LLM into a chain
chain = prompt | llm

# File uploader
upload_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
question = st.text_input("Ask a question about the image:")

# Process when user submits
if question:
    if upload_file is None:
        st.error("Please upload an image first!")
        st.stop()

    image_b64 = encode_image(upload_file)

    with st.spinner("Analyzing the image..."):
        response = chain.invoke({"input": question, "image": image_b64})

    # Display image and model response
    st.image(upload_file, caption="Uploaded Image", use_column_width=True)
    st.subheader("AI Response:")
    st.write(response.content)














