from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os   
import streamlit as st
load_dotenv()



def get_embeddings():
    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-base-en-v1.5",
        task="feature-extraction",
        huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"],
    )
    return embeddings
