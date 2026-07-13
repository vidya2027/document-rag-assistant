"""Command-line entry point for the Document RAG Assistant."""

from src.rag_pipeline import initialize_pipeline, rag_pipeline


def main():
    """Build the document index and start an interactive question loop."""
    try:
        state = initialize_pipeline(rebuild_index=True)
    except Exception as error:
        print(f"\nStartup error: {error}")
        return

    print("\nDocument RAG Assistant is ready.")
    print("Ask about files in the documents/ folder. Type 'exit' to stop.")

    while True:
        query = input("\nQuestion: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not query:
            continue

        try:
            result = rag_pipeline(state, query)

            print(f"\nAnswer: {result['answer']}")
            print(f"Sources: {result['sources']}")

            print("\nRetrieved evidence:")
            for index, (text, distance, metadata) in enumerate(
                zip(
                    result["retrieved_chunks"],
                    result["distances"],
                    result["metadatas"],
                ),
                start=1,
            ):
                print(
                    f"\n[{index}] distance={distance:.4f} "
                    f"source={metadata['source']} page={metadata['page']}"
                )
                print(text[:300])

        except Exception as error:
            print(f"Pipeline error: {error}")


if __name__ == "__main__":
    main()
