"""Local semantic embedding generation."""

from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name):
    """Load a Sentence Transformer embedding model."""
    return SentenceTransformer(model_name)


def get_embeddings(model, texts):
    """Generate normalized local embeddings as Python lists."""
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return vectors.tolist()
