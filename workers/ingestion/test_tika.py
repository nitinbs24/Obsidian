"""
test_tika.py — Quick smoke test for TikaExtractor.

Usage:
    python test_tika.py                          # tests with a dummy .txt file
    python test_tika.py /mnt/e/path/to/file.pdf  # tests with a real file
"""

import asyncio
import sys
from pathlib import Path

from tika_extractor import TikaExtractor


async def main():
    extractor = TikaExtractor(tika_url="http://localhost:9998")

    # --- 1. Health check ---
    print("Checking Tika server health...")
    healthy = await extractor.health_check()
    if not healthy:
        print("❌ Tika server is NOT reachable at http://localhost:9998")
        print("   Make sure Docker is running: docker compose up -d")
        return
    print("✅ Tika server is healthy\n")

    # --- 2. Pick a file ---
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return
        file_bytes = file_path.read_bytes()
        content_type = "application/pdf" if file_path.suffix == ".pdf" else "text/plain"
    else:
        # Fallback: use a small inline text sample
        print("No file provided — using inline text sample.\n")
        file_bytes = b"Hello from Tika! This is a test document with some sample content."
        content_type = "text/plain"

    # --- 3. Extract ---
    print(f"Extracting from: {sys.argv[1] if len(sys.argv) > 1 else '<inline text>'}")
    print(f"Content-Type:    {content_type}\n")

    result = await extractor.extract(file_bytes, content_type=content_type)

    # --- 4. Print results ---
    if not result.success:
        print(f"❌ Extraction failed: {result.error}")
        return

    print("=" * 60)
    print("EXTRACTED TEXT (first 500 chars):")
    print("=" * 60)
    print(result.text[:500] if result.text else "<empty>")
    print()
    print("=" * 60)
    print("METADATA:")
    print("=" * 60)
    for key, value in list(result.metadata.items())[:15]:  # show first 15 fields
        print(f"  {key}: {value}")

    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
