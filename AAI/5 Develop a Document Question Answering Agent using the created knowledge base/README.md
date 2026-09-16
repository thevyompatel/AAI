# Campus Connect Document Q&A Agent

A small Retrieval-Augmented Generation (RAG) agent built with LangChain, local Hugging Face embeddings, FAISS/Chroma, and Google Gemini.

## Quick start

1. Create a Python 3.13 virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and add `GOOGLE_API_KEY`.
4. Place your own `.txt` or `.pdf` knowledge files in `data/`.
5. Run `python build_index.py` once after changing source documents.
6. Run `python app.py --store faiss` for interactive Q&A.
7. Or ask one question directly with `python app.py --store faiss --query "What are the library hours?"`.

## Example questions

- What are the library opening hours?
- How many books can a student borrow?
- How long is a computer lab reservation?
- How far in advance should a club book a seminar hall?
- Does the shuttle require a reservation?
- What does the technical support desk handle?
- What is the policy for personal hardware repair?
