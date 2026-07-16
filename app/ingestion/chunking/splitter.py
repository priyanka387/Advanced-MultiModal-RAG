from typing import List
import logfire


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> List[str]:
    """
    Robust chunker.

    • Paragraph aware
    • Splits long paragraphs
    • Safe overlap
    • Never infinite loops
    """

    with logfire.span("Chunking", length=len(text)):

        if not text.strip():
            return []

        paragraphs = [
            p.strip()
            for p in text.replace("\r", "").split("\n\n")
            if p.strip()
        ]

        chunks = []
        current = ""

        for para in paragraphs:

            # Long paragraph
            if len(para) > chunk_size:

                if current:
                    chunks.append(current.strip())
                    current = ""

                start = 0

                while start < len(para):

                    end = min(start + chunk_size, len(para))

                    chunks.append(para[start:end])

                    # VERY IMPORTANT
                    if end >= len(para):
                        break

                    start = end - overlap

                continue

            # Normal paragraph
            if len(current) + len(para) + 2 <= chunk_size:
                current += para + "\n\n"

            else:

                if current.strip():
                    chunks.append(current.strip())

                current = para + "\n\n"

        if current.strip():
            chunks.append(current.strip())

        chunks = [c for c in chunks if c.strip()]

        logfire.info(f"Generated {len(chunks)} chunks")

        return chunks