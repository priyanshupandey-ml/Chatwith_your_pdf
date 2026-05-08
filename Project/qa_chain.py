from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
)


def create_qa_chain(vectorstore):

    # LLM
    chat_model = ChatHuggingFace(llm=llm)

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant.
    Answer the question using ONLY the provided context.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # Document chain
    document_chain = create_stuff_documents_chain(chat_model, prompt)

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10}
    )

    # Final RAG chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain