import os
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import base64


def encode_image(image_file):
 return base64.b64encode(image_file.read()).decode()



with st.sidebar:
 st.title("Provide API")
 OPENAI_API_KEY=st.text_input("OpenAI API key",type="password")

if not OPENAI_API_KEY:
 st.text("Enter API")
 st.stop()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedings = OpenAIEmbeddings(api_key="OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key="OPENAI_API_KEY")

prompt = ChatPromptTemplate.from_messages(
 [
 ("system", "You are a helpful assistant that can describe images."),
 (
 "human",
 [
 {"type": "text", "text": "{input}"},
 {
 "type": "image_url",
 "image_url": {
 "url": f"data:image/jpeg;base64,""{image}",
 "detail": "low",
 },
 },
 ],
 ),
 ]
)

chain=prompt | llm

upload_file=st.file_uploader("Upload the image",type=["jpg","png"])
question=st.text_input("Enter the question")

if question:
 image=encode_image(upload_file)
 
try:
    response=chain.invoke({"input":question,"image":image})
except Exception as e:
    print(f"An error occurred: {e}")
st.write(response.content)





