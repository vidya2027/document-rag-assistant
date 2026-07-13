"""Persistent ChromaDB vector storage and retrieval."""

import chromadb


def create_collection(chroma_path, collection_name, reset=False):
    """Create or retrieve a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=chroma_path)

    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    return client.get_or_create_collection(name=collection_name)


def index_chunks(collection, chunks, embeddings):
    """Store chunk text, metadata, IDs, and precomputed embeddings."""
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk and embedding counts do not match.")

    collection.add(
        ids=[f"chunk_{chunk['metadata']['chunk_id']}" for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
        embeddings=embeddings,
    )


def retrieve(collection, query_embedding, n_results=3):
    """Retrieve semantically similar chunks from ChromaDB."""
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "distances", "metadatas"],
    )

    return {
        "documents": result["documents"][0],
        "distances": result["distances"][0],
        "metadatas": result["metadatas"][0],
    }
