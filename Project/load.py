from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv
import os   
load_dotenv()

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    return chunks

def get_embeddings():
    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-base-en-v1.5",
        task="feature-extraction",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
    return embeddings

CHROMA_PATH = "chroma_db"

def create_vector_store(chunks, embeddings):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        # persist_directory=CHROMA_PATH
    )
    
    vectorstore.persist()
    return vectorstore


def load_vector_store(embeddings):
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore

documents = load_pdf("D:\Rag Project-1\Project\Daily Routine.pdf")
chunks = split_docs(documents)
embeddings = get_embeddings()
embeddings_list = embeddings.embed_documents([chunk.page_content for chunk in chunks])
vectorstore = create_vector_store(chunks, embeddings)
