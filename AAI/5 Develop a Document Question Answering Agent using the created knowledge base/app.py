import argparse
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

FAISS_INDEX_DIR = "vectorstore_faiss"
CHROMA_INDEX_DIR = "vectorstore_chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT_TEMPLATE = """You are Campus Connect Q&A Agent.
Answer the user's question using ONLY the retrieved context below.
If the answer is not present in the context, say: "I don't know based on the knowledge base."
Do not invent dates, fees, contact details, locations, policies, or other facts.
Keep the answer clear and concise.

Context:
{context}

Question: {question}
Answer:"""


def load_vectorstore(store: str, embeddings):
    """STEP 5: Load the previously built vector index."""
    if store == "faiss":
        return FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    if store == "chroma":
        return Chroma(
            collection_name="knowledge_base",
            embedding_function=embeddings,
            persist_directory=CHROMA_INDEX_DIR,
        )
    raise ValueError("store must be 'faiss' or 'chroma'")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever):
    """STEP 6: Connect retrieval -> prompt -> Gemini -> answer parser."""
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    parser = argparse.ArgumentParser(description="Campus Connect Document Q&A Agent")
    parser.add_argument("--store", choices=["faiss", "chroma"], default="faiss")
    parser.add_argument("--query", type=str, default=None, help="Ask one question and exit")
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve")
    args = parser.parse_args()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = load_vectorstore(args.store, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": args.k})
    chain = build_rag_chain(retriever)

    if args.query:
        print(chain.invoke(args.query))
        return

    print(f"Campus Connect Q&A Agent ready (store={args.store}). Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        answer = chain.invoke(question)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
