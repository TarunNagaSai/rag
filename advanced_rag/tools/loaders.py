"""Document loaders — PDF (column-aware) and plain text."""

from __future__ import annotations

from pathlib import Path

from advanced_rag.schema.schema import Document

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


def load_path(path: str | Path) -> list[Document]:
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
    if p.suffix.lower() == ".pdf":
        return _load_pdf(p)
    if p.suffix.lower() in TEXT_SUFFIXES:
        return [Document(text=p.read_text(encoding="utf-8", errors="ignore"), source=str(p))]
    raise ValueError(f"Unsupported file type: {p}")


def load_text(text: str, source: str = "inline") -> list[Document]:
    return [Document(text=text, source=source)]


def _extract_page_text(page) -> str:
    """Column-aware extraction: detects two-column layouts by measuring the gap
    between the rightmost word of the left half and the leftmost of the right
    half. When a clear gap exists, each column is cropped and extracted
    independently so text from adjacent columns is never interleaved."""
    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    if not words:
        return ""

    mid = page.width / 2
    left_xs = [w["x1"] for w in words if w["x0"] < mid]
    right_xs = [w["x0"] for w in words if w["x0"] >= mid]

    if left_xs and right_xs and (min(right_xs) - max(left_xs)) >= 10:
        left_text = (
            page.crop((0, 0, max(left_xs) + 5, page.height))
            .extract_text(x_tolerance=3, y_tolerance=3) or ""
        )
        right_text = (
            page.crop((min(right_xs) - 5, 0, page.width, page.height))
            .extract_text(x_tolerance=3, y_tolerance=3) or ""
        )
        return left_text + ("\n\n" + right_text if right_text.strip() else "")

    return page.extract_text(x_tolerance=3, y_tolerance=3) or ""


def _load_pdf(path: Path) -> list[Document]:
    import pdfplumber

    docs: list[Document] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = _extract_page_text(page)
            if text.strip():
                docs.append(
                    Document(text=text, source=str(path), metadata={"page": i + 1})
                )
    return docs
