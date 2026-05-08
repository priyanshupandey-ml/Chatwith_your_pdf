from dotenv import load_dotenv
load_dotenv()

from embeddings import get_embeddings
from vector_store import load_vector_store
from qa_chain import create_qa_chain

def main():
    print("📄 Chat with your PDF (type 'exit' to quit)\n")

    # Load embeddings
    embeddings = get_embeddings()

    # Load existing Chroma DB
    vectorstore = load_vector_store(embeddings)

    # Create RAG chain
    qa_chain = create_qa_chain(vectorstore)

    # Chat loop
    while True:
        question = input("\nAsk question: ")

        if question.lower() in ["exit", "quit"]:
            print("Goodbye 👋")
            break

        # NEW retrieval chain call
        response = qa_chain.invoke({"input": question})

        # The chain returns a dictionary
        answer = response["answer"]

        print("\n🤖 Answer:")
        print(answer)

if __name__ == "__main__":
    main()