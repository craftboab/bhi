import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
)
# import openai

# import os 
# openai.api_key = st.secrets.OpenAIAPI.openai_api_key
openai_api_key = st.secrets.OpenAIAPI.openai_api_key

# Streamlit Community Cloudの「Secrets」からOpenAI API keyを取得
# openai.api_key = st.secrets.OpenAIAPI.openai_api_key

# Load the document
loader = TextLoader("./test.txt", encoding="utf8")

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=0,
    length_function=len,
)

# Create the vector index
index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=OpenAIEmbeddings(openai_api_key=openai_api_key),
    text_splitter=text_splitter,
).from_loaders([loader])

chat = ChatOpenAI(temperature=0)


def chat_start(prompt):
  answer = index.query(prompt)
  response = chat([
    SystemMessage(content="あなたはロサンゼルスを中心に活躍する敏腕留学エージェントです。受け取った文字列を日本語に変換して留学エージェントとしてふさわしい丁寧な言い回しにしてください。尚、留学以外の質問に対してはお断りしてください。"),
    HumanMessage(content=answer),
  ])
  
  return response

# Streamlit application code
st.title('ロサンゼルス留学エージェント')
st.caption("by boab")

user_input = st.text_input('留学に関する質問を投げかけてください。')
if st.button('Start Chat'):
    with st.spinner('Agent is typing...'):
        st.text(chat_start(user_input))
