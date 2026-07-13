"""Text cleaning and recursive chunking."""

import re
import unicodedata


def clean_text(text):
    """Normalize Unicode, remove junk characters, and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", " ").replace("\ufffd", " ")
    text = re.sub(r"[^\w\s.,!?;:'\"()\-/%]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def recursive_chunk(text, chunk_size=500):
    """Chunk text by sentence boundaries before falling back to characters."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        candidate = f"{current} {sentence}".strip()

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(sentence) <= chunk_size:
            current = sentence
        else:
            chunks.extend(
                sentence[start:start + chunk_size]
                for start in range(0, len(sentence), chunk_size)
            )
            current = ""

    if current:
        chunks.append(current)

    return chunks


def prepare_chunks(documents, chunk_size=500):
    """Clean documents, chunk them, and attach unique chunk metadata."""
    chunks = []

    for document in documents:
        cleaned = clean_text(document["content"])

        for chunk_text in recursive_chunk(cleaned, chunk_size):
            chunk_id = len(chunks)
            metadata = document["metadata"].copy()
            metadata["chunk_id"] = chunk_id

            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": metadata,
                }
            )

    return chunks
