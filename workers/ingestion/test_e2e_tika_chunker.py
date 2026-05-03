"""
test_e2e_tika_chunker.py — End-to-end test: Tika extraction → Text chunking.

Tests the full pipeline from raw file bytes through to chunked output.
Requires Tika to be running: docker compose up -d

Usage:
    python test_e2e_tika_chunker.py                          # uses inline text sample
    python test_e2e_tika_chunker.py /path/to/file.pdf        # uses a real file
"""

import asyncio
import sys
from pathlib import Path

from tika_extractor import TikaExtractor, is_supported
from chunker import TextChunker


# ---------------------------------------------------------------------------
# MIME type lookup
# ---------------------------------------------------------------------------

EXTENSION_TO_MIME = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
}


async def main():
    extractor = TikaExtractor(tika_url="http://localhost:9998")
    chunker = TextChunker(chunk_size=512, chunk_overlap=50)

    # ── Step 0: Health check ──────────────────────────────────────────
    print("=" * 60)
    print("End-to-End Test: Tika → Chunker")
    print("=" * 60)

    print("\n[1/4] Checking Tika server health...")
    healthy = await extractor.health_check()
    if not healthy:
        print("❌ Tika server is NOT reachable at http://localhost:9998")
        print("   Run: docker compose up -d")
        return
    print("✅ Tika server is healthy")

    # ── Step 1: Load file ─────────────────────────────────────────────
    print("\n[2/4] Loading file...")
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return
        file_bytes = file_path.read_bytes()
        suffix = file_path.suffix.lower()
        content_type = EXTENSION_TO_MIME.get(suffix, "application/octet-stream")
        source_name = file_path.name

        if not is_supported(content_type):
            print(f"⚠️  MIME type '{content_type}' not in supported list, attempting anyway...")
    else:
        print("No file provided — using inline text sample.\n")
        sample_text = (
            "Apache Tika is a content detection and analysis framework.\n\n"
            "It provides a simple API for detecting the MIME type of a document "
            "and extracting text and metadata. Tika supports over 1,400 file formats "
            "including PDF, Microsoft Office, OpenDocument, and many more.\n\n"
            "The ingestion pipeline uses Tika as the first stage to convert raw "
            "uploaded files into plain text. This text is then split into overlapping "
            "chunks by the TextChunker, which uses LangChain's "
            "RecursiveCharacterTextSplitter under the hood.\n\n"
            "Each chunk carries positional metadata — chunk_index, start_char, "
            "end_char, and total_chunks — so that downstream systems can reconstruct "
            "the original context or highlight matched passages in search results.\n\n"
            "The chunked text is then sent to the model server for vector embedding "
            "using all-MiniLM-L6-v2, which produces 384-dimensional vectors. These "
            "vectors are stored in OpenSearch alongside the raw text for hybrid "
            "retrieval combining BM25 keyword search with semantic vector similarity."
        )
        file_bytes = sample_text.encode("utf-8")
        content_type = "text/plain"
        source_name = "<inline sample>"

    print(f"   Source:       {source_name}")
    print(f"   Size:         {len(file_bytes):,} bytes")
    print(f"   Content-Type: {content_type}")

    # ── Step 2: Extract via Tika ──────────────────────────────────────
    print("\n[3/4] Extracting text via Tika...")
    result = await extractor.extract(file_bytes, content_type=content_type)

    if not result.success:
        print(f"❌ Extraction failed: {result.error}")
        return

    if result.is_empty:
        print("⚠️  Tika returned empty text — nothing to chunk")
        return

    print(f"✅ Extracted {len(result.text):,} characters")

    if result.metadata:
        print(f"   Metadata fields: {len(result.metadata)}")
        # Show a few interesting metadata fields
        for key in ["Content-Type", "dc:title", "pdf:PDFVersion", "xmpTPg:NPages"]:
            if key in result.metadata:
                print(f"   {key}: {result.metadata[key]}")

    # ── Step 3: Chunk the extracted text ──────────────────────────────
    print("\n[4/4] Chunking extracted text...")
    chunks = chunker.chunk(result.text)

    if not chunks:
        print("⚠️  Chunker returned no chunks (text may be too short)")
        return

    print(f"✅ Produced {len(chunks)} chunks\n")

    # ── Results ───────────────────────────────────────────────────────
    print("=" * 60)
    print("CHUNK DETAILS")
    print("=" * 60)

    for c in chunks:
        print(f"\n── Chunk {c.chunk_index + 1}/{c.total_chunks} "
              f"(chars {c.start_char}–{c.end_char}, {len(c.text)} chars) ──")
        # Show first 200 chars of each chunk
        preview = c.text[:200]
        if len(c.text) > 200:
            preview += "..."
        print(preview)

    # ── Offset verification ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OFFSET VERIFICATION")
    print("=" * 60)

    stripped = result.text.strip()
    all_ok = True
    for c in chunks:
        extracted = stripped[c.start_char:c.end_char]
        if extracted != c.text:
            print(f"❌ Chunk {c.chunk_index}: offset mismatch!")
            all_ok = False

    if all_ok:
        print(f"✅ All {len(chunks)} chunk offsets verified against original text")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"   Source:          {source_name}")
    print(f"   Raw bytes:       {len(file_bytes):,}")
    print(f"   Extracted chars: {len(result.text):,}")
    print(f"   Chunks:          {len(chunks)}")
    print(f"   Chunk size:      {chunker.chunk_size}")
    print(f"   Overlap:         {chunker.chunk_overlap}")
    print(f"   Offsets valid:   {'✅ Yes' if all_ok else '❌ No'}")
    print("\n✅ End-to-end test complete!")


if __name__ == "__main__":
    asyncio.run(main())
