from pdfloader import load_pdf
from text_splitter import split_docs
from embeddings import get_embeddings
from vector_store import create_vector_store

documents = load_pdf("D:\Rag Project-1\Project\Daily Routine.pdf")
chunks = split_docs(documents)
embeddings = get_embeddings()

create_vector_store(chunks, embeddings)

print("PDF processed successfully!")