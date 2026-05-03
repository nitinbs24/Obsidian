"""
test_chunker.py — Tests for the TextChunker module.

Usage:
    python test_chunker.py

Runs a suite of edge-case tests and prints pass/fail results.
No external services required — only needs langchain-text-splitters.
"""

from chunker import TextChunker, ChunkResult


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_pass_count = 0
_fail_count = 0


def _check(name: str, condition: bool, detail: str = "") -> None:
    """Print a pass/fail line and update counters."""
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        print(f"  ✅ {name}")
    else:
        _fail_count += 1
        msg = f"  ❌ {name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_empty_string():
    """Empty input returns empty list."""
    print("\n--- test_empty_string ---")
    chunker = TextChunker()
    result = chunker.chunk("")
    _check("returns empty list", result == [])


def test_whitespace_only():
    """Whitespace-only input returns empty list."""
    print("\n--- test_whitespace_only ---")
    chunker = TextChunker()
    result = chunker.chunk("   \n\n\t  ")
    _check("returns empty list", result == [])


def test_below_min_chunk_size():
    """Text shorter than min_chunk_size returns empty list."""
    print("\n--- test_below_min_chunk_size ---")
    chunker = TextChunker(min_chunk_size=50)
    result = chunker.chunk("Short text.")
    _check("returns empty list", result == [], f"got {len(result)} chunks")


def test_single_chunk():
    """Text shorter than chunk_size returns exactly one chunk."""
    print("\n--- test_single_chunk ---")
    text = "This is a small piece of text that fits in a single chunk."
    chunker = TextChunker(chunk_size=512)
    result = chunker.chunk(text)
    _check("exactly 1 chunk", len(result) == 1, f"got {len(result)}")
    if result:
        c = result[0]
        _check("chunk_index == 0", c.chunk_index == 0)
        _check("total_chunks == 1", c.total_chunks == 1)
        _check("start_char == 0", c.start_char == 0)
        _check("end_char == len(text)", c.end_char == len(text.strip()))
        _check("text matches input", c.text == text.strip())


def test_multi_chunk():
    """Long text produces multiple chunks with sequential indices."""
    print("\n--- test_multi_chunk ---")
    # Build a ~2000 char text with clear paragraph breaks
    paragraphs = []
    for i in range(10):
        paragraphs.append(
            f"Paragraph {i}. " + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."
        )
    text = "\n\n".join(paragraphs)

    chunker = TextChunker(chunk_size=200, chunk_overlap=30)
    result = chunker.chunk(text)

    _check("multiple chunks", len(result) > 1, f"got {len(result)}")

    # Check sequential indices
    indices = [c.chunk_index for c in result]
    _check("indices are sequential", indices == list(range(len(result))))

    # Check total_chunks consistency
    totals = {c.total_chunks for c in result}
    _check("all total_chunks agree", len(totals) == 1 and totals.pop() == len(result))


def test_offset_correctness():
    """For every chunk, original_text[start_char:end_char] matches chunk.text."""
    print("\n--- test_offset_correctness ---")
    paragraphs = [
        "First paragraph with some content about search engines and indexing.",
        "Second paragraph discussing vector embeddings and similarity search.",
        "Third paragraph about Apache Tika and text extraction from documents.",
        "Fourth paragraph covering Kafka event streaming and message queues.",
        "Fifth paragraph on OpenSearch and hybrid retrieval mechanisms.",
    ]
    text = "\n\n".join(paragraphs)

    chunker = TextChunker(chunk_size=150, chunk_overlap=20)
    result = chunker.chunk(text)

    stripped = text.strip()
    all_match = True
    for c in result:
        extracted = stripped[c.start_char:c.end_char]
        if extracted != c.text:
            all_match = False
            print(f"    MISMATCH at chunk {c.chunk_index}:")
            print(f"      expected: {c.text[:60]}...")
            print(f"      got:      {extracted[:60]}...")
            break

    _check(
        "all offsets match original text",
        all_match,
        f"checked {len(result)} chunks",
    )


def test_custom_parameters():
    """Custom chunk_size and overlap produce expected behaviour."""
    print("\n--- test_custom_parameters ---")
    text = "word " * 200  # 1000 chars
    chunker = TextChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=10)
    result = chunker.chunk(text)

    _check("multiple chunks with small chunk_size", len(result) > 5, f"got {len(result)}")

    # Each chunk should be <= chunk_size
    all_within_limit = all(len(c.text) <= 100 for c in result)
    _check("all chunks ≤ chunk_size", all_within_limit)


def test_min_chunk_filter():
    """Tiny fragments produced by splitting are filtered out."""
    print("\n--- test_min_chunk_filter ---")
    # Create text that could produce very short chunks at boundaries
    chunker = TextChunker(chunk_size=100, chunk_overlap=10, min_chunk_size=30)
    text = "A" * 95 + "\n\n" + "B" * 5 + "\n\n" + "C" * 95
    result = chunker.chunk(text)

    # The "BBBBB" fragment (5 chars) should be filtered out
    short_chunks = [c for c in result if len(c.text) < 30]
    _check(
        "no chunks below min_chunk_size",
        len(short_chunks) == 0,
        f"found {len(short_chunks)} short chunks",
    )


def test_chunk_result_fields():
    """ChunkResult has all expected fields."""
    print("\n--- test_chunk_result_fields ---")
    cr = ChunkResult(
        text="test", chunk_index=0, start_char=0, end_char=4, total_chunks=1
    )
    _check("has text", hasattr(cr, "text"))
    _check("has chunk_index", hasattr(cr, "chunk_index"))
    _check("has start_char", hasattr(cr, "start_char"))
    _check("has end_char", hasattr(cr, "end_char"))
    _check("has total_chunks", hasattr(cr, "total_chunks"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TextChunker — Test Suite")
    print("=" * 60)

    test_empty_string()
    test_whitespace_only()
    test_below_min_chunk_size()
    test_single_chunk()
    test_multi_chunk()
    test_offset_correctness()
    test_custom_parameters()
    test_min_chunk_filter()
    test_chunk_result_fields()

    print("\n" + "=" * 60)
    print(f"Results: {_pass_count} passed, {_fail_count} failed")
    print("=" * 60)

    if _fail_count > 0:
        print("\n❌ Some tests failed!")
        exit(1)
    else:
        print("\n✅ All tests passed!")
