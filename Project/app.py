import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from embeddings import get_embeddings
from vector_store import create_vector_store, load_vector_store
from qa_chain import create_qa_chain
from pdfloader import load_pdf
from text_splitter import split_docs

st.set_page_config(page_title="Chat with PDF", page_icon="📄")
st.title("📄 Chat with your PDF")

# Session state to keep chain loaded
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if st.button("Process PDF"):
        if uploaded_file is None:
            st.warning("Please upload a PDF first!")
        else:
            with st.spinner("Processing PDF..."):
                # Save uploaded file temporarily
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.read())

                # Load + split PDF
                documents = load_pdf("temp.pdf")
                chunks = split_docs(documents)

                # Create embeddings
                embeddings = get_embeddings()

                # Create vector database
                vectorstore = create_vector_store(chunks, embeddings)

                # Create RAG chain
                st.session_state.qa_chain = create_qa_chain(vectorstore)

                st.success("PDF processed! You can now ask questions.")

# Chat UI
question = st.text_input("Ask a question from the PDF")

if question and st.session_state.qa_chain:
    with st.spinner("Thinking..."):
        response = st.session_state.qa_chain.invoke({"input": question})
        st.write("🤖 **Answer:**")
        st.write(response["answer"])

elif question and st.session_state.qa_chain is None:
    st.warning("Please upload and process a PDF first.")