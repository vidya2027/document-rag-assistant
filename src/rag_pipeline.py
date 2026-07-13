"""End-to-end Document RAG Assistant pipeline."""

from src.config import (
    HF_TOKEN,
    EMBEDDING_MODEL,
    LLM_MODEL,
    DOCUMENTS_DIR,
    CHROMA_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    RETRIEVAL_K,
)
from src.document_loader import load_documents_from_folder
from src.text_processing import prepare_chunks
from src.embeddings import load_embedding_model, get_embeddings
from src.vector_store import create_collection, index_chunks, retrieve
from src.llm import create_hf_client, ask_llm

def build_index():
    """Load the documents folder and rebuild the persistent vector index."""
    print(f"Reading documents from: {DOCUMENTS_DIR}")
    documents = load_documents_from_folder(DOCUMENTS_DIR)
    chunks = prepare_chunks(documents, CHUNK_SIZE)

    print(f"Loaded {len(documents)} document units.")
    print(f"Created {len(chunks)} chunks.")

    embedding_model = load_embedding_model(EMBEDDING_MODEL)
    embeddings = get_embeddings(
        embedding_model,
        [chunk["content"] for chunk in chunks],
    )

    collection = create_collection(
        CHROMA_PATH,
        COLLECTION_NAME,
        reset=True,
    )
    index_chunks(collection, chunks, embeddings)

    print(f"Stored {collection.count()} chunks in ChromaDB.")
    return embedding_model, collection


def initialize_pipeline(rebuild_index=True):
    """Initialize embeddings, ChromaDB, and authenticated HF inference."""
    if rebuild_index:
        embedding_model, collection = build_index()
    else:
        embedding_model = load_embedding_model(EMBEDDING_MODEL)
        collection = create_collection(
            CHROMA_PATH,
            COLLECTION_NAME,
            reset=False,
        )

        if collection.count() == 0:
            raise ValueError("ChromaDB index is empty. Run with rebuild_index=True.")

    hf_client = create_hf_client(HF_TOKEN)
    return {
        "embedding_model": embedding_model,
        "collection": collection,
        "hf_client": hf_client,
    }


def rag_pipeline(state, query):
    """Retrieve relevant chunks and generate a grounded answer."""
    query_embedding = get_embeddings(
        state["embedding_model"],
        [query],
    )[0]

    results = retrieve(
        state["collection"],
        query_embedding,
        RETRIEVAL_K,
    )

    answer = ask_llm(
        state["hf_client"],
        LLM_MODEL,
        query,
        results["documents"],
    )

    sources = sorted(
        {
            metadata["source"]
            for metadata in results["metadatas"]
        }
    )

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": results["documents"],
        "distances": results["distances"],
        "metadatas": results["metadatas"],
    }
