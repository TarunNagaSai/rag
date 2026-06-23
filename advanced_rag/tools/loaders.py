"""Document loaders. Small on purpose — add formats as you need them.

Loaders only produce raw ``Document`` objects. All the interesting structure-aware
work happens in ``chunking.py`` so loaders stay trivially testable.
"""

from __future__ import annotations

from pathlib import Path

from .schema import Document

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


def load_path(path: str | Path) -> list[Document]:
    """Load a file or a directory (recursively) into Documents."""
    p = Path(path)
    if p.is_dir():
        docs: list[Document] = []
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix.lower() in TEXT_SUFFIXES | {".pdf"}:
                docs.extend(load_file(f))
        return docs
    return load_file(p)


def load_file(path: str | Path) -> list[Document]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(p)
    if suffix in TEXT_SUFFIXES:
        return [Document(text=p.read_text(encoding="utf-8", errors="ignore"), source=str(p))]
    raise ValueError(f"Unsupported file type: {p}")


def _load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs: list[Document] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(text=text, source=str(path), metadata={"page": i + 1})
            )
    return docs


def load_text(text: str, source: str = "inline") -> list[Document]:
    return [Document(text=text, source=source)]
