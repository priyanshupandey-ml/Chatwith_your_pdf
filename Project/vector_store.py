from langchain_community.vectorstores import Chroma



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