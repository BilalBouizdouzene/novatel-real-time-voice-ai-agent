"""Markdown loading and paragraph-aware character chunking for the demo corpus."""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    source: str
    source_url: str | None = None
    retrieved_at: str | None = None


def clean_document(text: str) -> str:
    """Normalize whitespace while retaining paragraph boundaries."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_provenance(text: str) -> tuple[str | None, str | None]:
    """Read optional source URL and collection timestamp from scraper metadata."""

    url_match = re.search(r"(?im)^\s*(?:url source|source url)\s*:\s*(https?://\S+)\s*$", text)
    date_match = re.search(r"(?im)^\s*(?:dernière collecte|last collected)\s*:\s*(.+?)\s*$", text)
    return (url_match.group(1) if url_match else None, date_match.group(1) if date_match else None)


def remove_provenance_lines(text: str) -> str:
    """Keep scraper bookkeeping out of the passages sent to the LLM."""

    return re.sub(r"(?im)^\s*(?:url source|source url|dernière collecte|last collected)\s*:.*(?:\n|$)", "", text)


def chunk_text(text: str, source: str, chunk_size: int = 1000, overlap: int = 150) -> list[DocumentChunk]:
    """Split text into bounded overlapping chunks, preferring paragraph edges."""

    cleaned = clean_document(text)
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    chunks: list[DocumentChunk] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            boundary = cleaned.rfind("\n", start + int(chunk_size * 0.65), end)
            if boundary > start:
                end = boundary
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(DocumentChunk(text=piece, source=source))
        if end >= len(cleaned):
            break
        start = max(start + 1, end - overlap)
    return chunks


def load_markdown_documents(directory: Path, chunk_size: int = 1000, overlap: int = 150) -> list[DocumentChunk]:
    """Read and chunk Markdown files using paths relative to the knowledge base."""

    if not directory.exists():
        return []
    chunks: list[DocumentChunk] = []
    for path in sorted(directory.rglob("*.md")):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        source_url, retrieved_at = extract_provenance(content)
        source = path.relative_to(directory).as_posix()
        searchable_content = remove_provenance_lines(content)
        chunks.extend(
            DocumentChunk(chunk.text, source, source_url, retrieved_at)
            for chunk in chunk_text(searchable_content, source, chunk_size=chunk_size, overlap=overlap)
        )
    return chunks
