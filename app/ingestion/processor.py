import os
import sys
import uuid
import json
import time
import traceback
import logfire
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings

from app.services.retrieval.embedding import (
    embed_texts,
    get_embedding_dim,
    get_embedding_backend,
)

from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.loaders.office import parse_office

from app.ingestion.chunking.splitter import chunk_text


logfire.configure(service_name="enterprise-ingestion-service")


PROCESSED_DATA_DIR = "processed_data"

# Ensure processed_data directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    timeout=60
)


# ==========================================================
# Save Processed JSON
# ==========================================================

def save_processed_locally(
    data: dict,
    source_type: str,
    filename: str,
) -> str:
    """Save processed document chunks to JSON file."""
    folder = os.path.join(
        PROCESSED_DATA_DIR,
        source_type,
    )

    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(
        folder,
        filename + ".json",
    )

    try:
        with open(
            filepath,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )

        logfire.info(f"Saved processed JSON: {filepath}")
        print(f"✓ Saved processed JSON -> {filepath}")

        return filepath

    except Exception as e:
        logfire.error(f"Failed to save processed JSON: {e}")
        raise


# ==========================================================
# Parse Document
# ==========================================================

def parse_document(
    file_path: str,
    extension: str,
) -> Optional[str]:
    """Parse document based on file extension."""
    try:
        if extension == "pdf":
            return parse_pdf(file_path)

        if extension in ("html", "htm"):
            return parse_html(file_path)

        if extension == "txt":
            return parse_text(file_path)

        if extension in ("docx", "pptx"):
            return parse_office(file_path)

        logfire.warning(f"Unsupported file extension: {extension}")
        return None

    except Exception as e:
        logfire.error(f"Error parsing document {file_path}: {e}")
        raise


# ==========================================================
# Process One File
# ==========================================================

def process_file(
    file_path: str,
    filename: str,
    source_type: str,
) -> bool:
    """Process a single file: parse, chunk, embed, and upload to Qdrant."""
    print("\n" + "=" * 70)
    print(f"Processing: {filename}")
    print("=" * 70)

    start_time = time.time()

    try:
        # Get file extension
        extension = filename.lower().split(".")[-1]

        # Parse document
        print("▸ Parsing document...")
        full_text = parse_document(file_path, extension)

        if full_text is None:
            logfire.warning(f"Unsupported file: {filename}")
            print("✗ Unsupported file.")
            return False

        full_text = full_text.strip()

        if not full_text:
            logfire.warning(f"No text extracted from: {filename}")
            print("✗ No text extracted.")
            return False

        # Chunk text
        print("▸ Chunking...")
        chunks = chunk_text(full_text)

        if len(chunks) == 0:
            logfire.warning(f"No chunks created from: {filename}")
            print("✗ No chunks created.")
            return False

        print(f"✓ Created {len(chunks)} chunks")

        # Save processed data locally
        processed = {
            "filename": filename,
            "source_type": source_type,
            "num_chunks": len(chunks),
            "chunks": chunks,
        }

        save_processed_locally(
            processed,
            source_type,
            filename,
        )

        # Generate embeddings
        print("▸ Generating embeddings...")
        embeddings = embed_texts(chunks)

        if len(embeddings) != len(chunks):
            error_msg = (
                f"Embedding count mismatch: "
                f"expected {len(chunks)}, got {len(embeddings)}"
            )
            logfire.error(error_msg)
            raise RuntimeError(error_msg)

        print(f"✓ Generated {len(embeddings)} embeddings")

        # Create Qdrant points
        print("▸ Creating Qdrant points...")
        points = []

        for chunk, vector in zip(chunks, embeddings):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source": filename,
                        "source_type": source_type,
                    },
                )
            )

        # Upload to Qdrant
        print("▸ Uploading to Qdrant...")
        qdrant_client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
        )

        elapsed = round(time.time() - start_time, 2)

        print("=" * 70)
        print(f"✓ SUCCESS: {filename}")
        print(f"  Vectors: {len(points)}")
        print(f"  Time: {elapsed}s")
        print("=" * 70)

        logfire.info(
            f"Successfully processed {filename}",
            extra={
                "file": filename,
                "chunks": len(chunks),
                "vectors": len(points),
                "time_sec": elapsed,
            }
        )

        return True

    except Exception as e:
        elapsed = round(time.time() - start_time, 2)

        print()
        print("=" * 70)
        print("✗ FAILED")
        print(filename)
        print("=" * 70)
        print(f"Error: {str(e)}")
        print("=" * 70)

        logfire.error(
            f"Failed to process {filename}",
            extra={
                "file": filename,
                "error": str(e),
                "time_sec": elapsed,
            }
        )

        traceback.print_exc()

        return False


# ==========================================================
# Process Directory
# ==========================================================

def process_directory(
    dir_path: str,
    source_type: str,
) -> dict:
    """Process all files in a directory."""
    if not os.path.isdir(dir_path):
        print(f"✗ Directory not found: {dir_path}")
        logfire.error(f"Directory not found: {dir_path}")
        return {"success": 0, "failed": 0, "skipped": 0, "total": 0}

    files = sorted([
        f
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
    ])

    total = len(files)
    success = 0
    failed = 0
    skipped = 0

    print()
    print("=" * 80)
    print(f"Processing Folder: {dir_path}")
    print(f"Source Type: {source_type}")
    print(f"Files: {total}")
    print("=" * 80)

    if total == 0:
        print("No files found in directory.")
        return {"success": 0, "failed": 0, "skipped": 0, "total": 0}

    for index, filename in enumerate(files, start=1):
        # Skip hidden files and system files
        if filename.startswith("."):
            skipped += 1
            continue

        print(f"\n[{index}/{total}]")

        ok = process_file(
            os.path.join(dir_path, filename),
            filename,
            source_type,
        )

        if ok:
            success += 1
        else:
            failed += 1

    print()
    print("=" * 80)
    print("DIRECTORY SUMMARY")
    print("=" * 80)
    print(f"Total:     {total}")
    print(f"Processed: {success}")
    print(f"Failed:    {failed}")
    print(f"Skipped:   {skipped}")
    print("=" * 80)

    logfire.info(
        f"Directory processing complete: {dir_path}",
        extra={
            "source_type": source_type,
            "total": total,
            "success": success,
            "failed": failed,
            "skipped": skipped,
        }
    )

    return {
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "total": total,
    }


# ==========================================================
# Universal Ingestion
# ==========================================================

def run_universal_ingestion(
    base_dir: str,
    explicit_source_type: Optional[str] = None,
    wipe: bool = False,
) -> dict:
    """
    Run universal ingestion pipeline.
    
    Args:
        base_dir: Base directory containing data files or subdirectories
        explicit_source_type: Explicit source type label (optional)
        wipe: Whether to delete existing collection
        
    Returns:
        Summary dict with processing statistics
    """
    print("\n")
    print("=" * 90)
    print("ENTERPRISE RAG INGESTION")
    print("=" * 90)

    # Get embedding backend info
    backend = get_embedding_backend()
    print(f"Embedding Backend: {backend}")

    embedding_dim = get_embedding_dim()
    print(f"Vector Dimension:  {embedding_dim}")

    print(f"Collection:        {settings.QDRANT_COLLECTION}")
    print("=" * 90)

    # Validate base directory
    if not os.path.isdir(base_dir):
        print(f"✗ Base directory not found: {base_dir}")
        logfire.error(f"Base directory not found: {base_dir}")
        return {"success": 0, "failed": 0, "total": 0}

    # Handle Qdrant collection deletion
    if wipe:
        print("\n▸ Deleting old collection...")
        try:
            if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                print("✓ Old collection removed.")
                logfire.info("Qdrant collection deleted.")
            else:
                print("✓ Collection did not exist.")
        except Exception as e:
            logfire.error(f"Failed to delete collection: {e}")
            traceback.print_exc()

    # Create collection if needed
    if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
        print("\n▸ Creating collection...")
        try:
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=embedding_dim,
                    distance=models.Distance.COSINE,
                ),
            )
            print("✓ Collection created.")
            logfire.info("Qdrant collection created.")
        except Exception as e:
            logfire.error(f"Failed to create collection: {e}")
            traceback.print_exc()
            return {"success": 0, "failed": 0, "total": 0}
    else:
        print("\n✓ Collection already exists.")

    # Process files
    print("\n" + "=" * 90)
    print("PROCESSING STARTED")
    print("=" * 90)

    subdirs = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    overall_stats = {
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "total": 0,
    }

    # If no subdirectories, process files directly
    if len(subdirs) == 0:
        if explicit_source_type:
            source_type = explicit_source_type
        else:
            folder_name = os.path.basename(os.path.normpath(base_dir)).lower()
            
            if "true" in folder_name:
                source_type = "true"
            elif "noisy" in folder_name:
                source_type = "noisy"
            else:
                source_type = "general"

        stats = process_directory(base_dir, source_type)
        overall_stats = stats

    else:
        # Process each subdirectory
        for folder in sorted(subdirs):
            if "true" in folder.lower():
                source_type = "true"
            elif "noisy" in folder.lower():
                source_type = "noisy"
            else:
                source_type = folder

            folder_path = os.path.join(base_dir, folder)
            stats = process_directory(folder_path, source_type)
            
            overall_stats["success"] += stats["success"]
            overall_stats["failed"] += stats["failed"]
            overall_stats["skipped"] += stats["skipped"]
            overall_stats["total"] += stats["total"]

    # Final summary
    print("\n")
    print("=" * 90)
    print("INGESTION PIPELINE COMPLETED")
    print("=" * 90)
    print(f"Total Files:     {overall_stats['total']}")
    print(f"Successfully:    {overall_stats['success']}")
    print(f"Failed:          {overall_stats['failed']}")
    print(f"Skipped:         {overall_stats['skipped']}")
    print(f"Processed Data:  {PROCESSED_DATA_DIR}/")
    print("=" * 90)

    logfire.info(
        "Ingestion pipeline completed",
        extra=overall_stats
    )

    return overall_stats


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    wipe_requested = "--wipe" in sys.argv

    args = [arg for arg in sys.argv if arg != "--wipe"]

    target_dir = args[1] if len(args) > 1 else "DATA"

    explicit_type = args[2] if len(args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"✗ Directory not found: {target_dir}")
        sys.exit(1)

    run_universal_ingestion(
        target_dir,
        explicit_source_type=explicit_type,
        wipe=wipe_requested,
    )