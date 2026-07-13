"""Load supported documents from a folder."""

from pathlib import Path
from PyPDF2 import PdfReader


def load_pdf(file_path):
    """Extract PDF text page by page and preserve source/page metadata."""
    documents = []
    reader = PdfReader(file_path)

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                {
                    "content": text,
                    "metadata": {
                        "source": Path(file_path).name,
                        "page": page_number,
                    },
                }
            )

    return documents


def load_txt(file_path):
    """Load one UTF-8 TXT file and preserve source metadata."""
    text = Path(file_path).read_text(encoding="utf-8", errors="replace")

    if not text.strip():
        return []

    return [
        {
            "content": text,
            "metadata": {
                "source": Path(file_path).name,
                "page": 1,
            },
        }
    ]


def load_documents_from_folder(folder_path):
    """Load every supported PDF and TXT file from a folder recursively."""
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Document folder does not exist: {folder}")

    documents = []
    supported_files = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt"}
    )

    if not supported_files:
        raise ValueError(
            f"No PDF or TXT documents found in '{folder}'. "
            "Add files to the documents folder and run again."
        )

    for file_path in supported_files:
        try:
            if file_path.suffix.lower() == ".pdf":
                loaded = load_pdf(file_path)
            else:
                loaded = load_txt(file_path)

            documents.extend(loaded)
            print(f"Loaded {file_path.name}: {len(loaded)} document unit(s)")
        except Exception as error:
            print(f"Skipped {file_path.name}: {error}")

    if not documents:
        raise ValueError("Files were found, but no readable text was extracted.")

    return documents
