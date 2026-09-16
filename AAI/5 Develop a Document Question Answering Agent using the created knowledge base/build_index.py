import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

DATA_DIR = "data"
FAISS_INDEX_DIR = "vectorstore_faiss"
CHROMA_INDEX_DIR = "vectorstore_chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    """STEP 1: Load every .txt and .pdf file found under data/."""
    docs = []

    txt_loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    docs.extend(pdf_loader.load())

    print(f"Loaded {len(docs)} raw document(s) from '{DATA_DIR}/'")
    return docs


def split_documents(docs):
    """STEP 2: Break documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunk(s)")
    return chunks


def build_faiss_index(chunks, embeddings):
    """STEP 4a: Create and save a FAISS vector index."""
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_DIR)
    print(f"FAISS index saved to '{FAISS_INDEX_DIR}/'")


def build_chroma_index(chunks, embeddings):
    """STEP 4b: Create and persist a Chroma collection."""
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name="knowledge_base",
        persist_directory=CHROMA_INDEX_DIR,
    )
    print(f"Chroma index saved to '{CHROMA_INDEX_DIR}/'")


def main():
    if not os.path.isdir(DATA_DIR) or not os.listdir(DATA_DIR):
        raise SystemExit(
            f"No files found in '{DATA_DIR}/'. Add a .txt or .pdf file there first."
        )

    docs = load_documents()
    chunks = split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    build_faiss_index(chunks, embeddings)
    build_chroma_index(chunks, embeddings)

    print("\nDone. Run: python app.py --store faiss")


if __name__ == "__main__":
    main()
