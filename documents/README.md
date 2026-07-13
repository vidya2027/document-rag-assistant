# Documents

This folder is the RAG knowledge base.

Three small, original sample documents are included so the project can run immediately:

- `rag_fundamentals.txt`
- `medical_imaging_ai.txt`
- `vector_search_notes.txt`

You can delete these files and add your own `.pdf` and `.txt` files. The loader searches this folder recursively, so subfolders are supported.

The application does **not** contain hardcoded fallback knowledge in Python. All indexed knowledge is loaded from files in this folder.

## Suggested demo questions

- What are the main stages of a RAG pipeline?
- Why is chunking important?
- What tasks can AI support in medical imaging?
- Why is human oversight important in healthcare AI?
- Why should the same embedding model be used for documents and queries?
- What happens when retrieval K is increased?
- Who won the 2018 FIFA World Cup?  
  This is intentionally outside the sample knowledge base and tests grounding behavior.
