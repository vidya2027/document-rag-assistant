"""Grounded prompt construction and Hugging Face LLM inference."""

from huggingface_hub import InferenceClient

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer questions ONLY based on the "
    "provided context. If the answer is not in the context, say "
    "\"I don't have enough information to answer this.\""
)


def create_hf_client(hf_token):
    """Create an authenticated Hugging Face inference client."""
    if not hf_token:
        raise ValueError(
            "HF_TOKEN is missing. Copy .env.example to .env and add your token."
        )

    return InferenceClient(
        provider="auto",
        api_key=hf_token,
        timeout=120,
    )


def build_prompt(query, retrieved_chunks):
    """Build a numbered, context-grounded RAG prompt."""
    context = "\n\n".join(
        f"[{index}] {text}"
        for index, text in enumerate(retrieved_chunks, start=1)
    )

    return f"""Context:
{context}

Question: {query}

Answer based only on the context above. If the context doesn't contain the answer, say "I don't have enough information to answer this."
"""


def ask_llm(client, model_name, query, retrieved_chunks):
    """Send retrieved context and a question to a Hugging Face chat model."""
    response = client.chat_completion(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_prompt(query, retrieved_chunks),
            },
        ],
        temperature=0.2,
        max_tokens=300,
    )

    return response.choices[0].message.content
