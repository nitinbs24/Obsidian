"""
test_model_client.py — Unit tests for ModelClient.

All tests mock httpx so no real model-server is needed.

Run:
    pytest workers/ingestion/test_model_client.py -v
"""

import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from model_client import ModelClient, EmbeddingError, _batchify


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """A ModelClient pointing at a fake server with tight retry settings."""
    return ModelClient(
        model_server_url="http://fake-model-server",
        timeout=5.0,
        batch_size=3,
        max_retries=2,
        backoff_base=0.01,   # near-zero backoff so tests run fast
    )


def _make_response(status: int, body: dict) -> MagicMock:
    """Build a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    resp.text = json.dumps(body)
    if status >= 400:
        from httpx import HTTPStatusError, Request
        resp.raise_for_status.side_effect = HTTPStatusError(
            message=f"HTTP {status}",
            request=MagicMock(spec=Request),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# _batchify helper
# ---------------------------------------------------------------------------

class TestBatchify:
    def test_exact_multiple(self):
        assert _batchify([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_remainder(self):
        assert _batchify([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

    def test_single_batch(self):
        assert _batchify([1, 2], 10) == [[1, 2]]

    def test_empty(self):
        assert _batchify([], 5) == []

    def test_size_one(self):
        assert _batchify([10, 20, 30], 1) == [[10], [20], [30]]


# ---------------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------------

class TestEmbedTexts:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self, client):
        result = await client.embed_texts([])
        assert result == []

    @pytest.mark.asyncio
    async def test_single_batch_success(self, client):
        fake_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_resp = _make_response(200, {"embeddings": fake_vectors})

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_http
            mock_http.request.return_value = mock_resp

            result = await client.embed_texts(["hello", "world"])

        assert result == fake_vectors

    @pytest.mark.asyncio
    async def test_multi_batch_reassembled_in_order(self, client):
        """batch_size=3, so 5 texts → [3] + [2]."""
        batch1_vecs = [[1.0] * 3, [2.0] * 3, [3.0] * 3]
        batch2_vecs = [[4.0] * 3, [5.0] * 3]
        responses = [
            _make_response(200, {"embeddings": batch1_vecs}),
            _make_response(200, {"embeddings": batch2_vecs}),
        ]

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_http
            mock_http.request.side_effect = responses

            result = await client.embed_texts(["a", "b", "c", "d", "e"])

        assert result == batch1_vecs + batch2_vecs
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_mismatched_vector_count_raises(self, client):
        """Server returns 1 vector for 2 texts — must raise ValueError."""
        mock_resp = _make_response(200, {"embeddings": [[0.1, 0.2]]})

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_http
            mock_http.request.return_value = mock_resp

            with pytest.raises(ValueError, match="expected 2 vectors"):
                await client.embed_texts(["hello", "world"])

    @pytest.mark.asyncio
    async def test_unexpected_body_raises_embedding_error(self, client):
        mock_resp = _make_response(200, {"wrong_key": []})

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_http
            mock_http.request.return_value = mock_resp

            with pytest.raises(EmbeddingError, match="unexpected body"):
                await client.embed_texts(["hello"])

    @pytest.mark.asyncio
    async def test_retries_on_5xx_then_succeeds(self, client):
        """First attempt gets 503, second succeeds."""
        import httpx as _httpx

        fail_resp = MagicMock()
        fail_resp.status_code = 503
        fail_resp.text = "Service Unavailable"
        fail_resp.raise_for_status.side_effect = _httpx.HTTPStatusError(
            message="503",
            request=MagicMock(spec=_httpx.Request),
            response=fail_resp,
        )

        ok_resp = _make_response(200, {"embeddings": [[0.1, 0.2]]})

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_http
            mock_http.request.side_effect = [fail_resp, ok_resp]

            result = await client.embed_texts(["hello"])

        assert result == [[0.1, 0.2]]
        assert mock_http.request.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_4xx(self, client):
        """4xx client errors should NOT be retried."""
        import httpx as _httpx

        bad_req_resp = MagicMock()
        bad_req_resp.status_code = 422
        bad_req_resp.text = "Unprocessable Entity"
        bad_req_resp.raise_for_status.side_effect = _httpx.HTTPStatusError(
            message="422",
            request=MagicMock(spec=_httpx.Request),
            response=bad_req_resp,
        )

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_http
            mock_http.request.return_value = bad_req_resp

            with pytest.raises(EmbeddingError, match="client error 422"):
                await client.embed_texts(["hello"])

        # Should have only tried once
        assert mock_http.request.call_count == 1

    @pytest.mark.asyncio
    async def test_raises_after_all_retries_exhausted(self, client):
        """All attempts time out → EmbeddingError."""
        import httpx as _httpx

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_http
            mock_http.request.side_effect = _httpx.TimeoutException("timed out")

            with pytest.raises(EmbeddingError, match="failed after 2 attempts"):
                await client.embed_texts(["hello"])

        assert mock_http.request.call_count == client.max_retries

    @pytest.mark.asyncio
    async def test_retries_on_connect_error(self, client):
        """ConnectError should be retried."""
        import httpx as _httpx

        ok_resp = _make_response(200, {"embeddings": [[0.9, 0.8]]})

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_http
            mock_http.request.side_effect = [
                _httpx.ConnectError("refused"),
                ok_resp,
            ]

            result = await client.embed_texts(["hi"])

        assert result == [[0.9, 0.8]]


# ---------------------------------------------------------------------------
# embed_image
# ---------------------------------------------------------------------------

class TestEmbedImage:
    @pytest.mark.asyncio
    async def test_empty_bytes_raises(self, client):
        with pytest.raises(ValueError, match="must not be empty"):
            await client.embed_image(b"")

    @pytest.mark.asyncio
    async def test_success(self, client):
        fake_vec = [0.11, 0.22, 0.33]
        mock_resp = _make_response(200, {"embedding": fake_vec})

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_http
            mock_http.request.return_value = mock_resp

            result = await client.embed_image(b"\xff\xd8\xff", "image/jpeg")

        assert result == fake_vec

    @pytest.mark.asyncio
    async def test_missing_embedding_key_raises(self, client):
        mock_resp = _make_response(200, {"result": []})

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_http
            mock_http.request.return_value = mock_resp

            with pytest.raises(EmbeddingError, match="unexpected body"):
                await client.embed_image(b"\x00\x01\x02")

    @pytest.mark.asyncio
    async def test_multipart_sent_correctly(self, client):
        """Verify the files= kwarg is used (not json=) for image embedding."""
        fake_vec = [0.5] * 384
        mock_resp = _make_response(200, {"embedding": fake_vec})

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_http
            mock_http.request.return_value = mock_resp

            await client.embed_image(b"\xff\xd8\xff", "image/png")

        call_kwargs = mock_http.request.call_args
        assert "files" in call_kwargs.kwargs
        assert call_kwargs.kwargs["files"]["file"][2] == "image/png"


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_returns_true_on_200(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_http
            mock_http.get.return_value = mock_resp

            assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_returns_false_on_error(self, client):
        import httpx as _httpx

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_http
            mock_http.get.side_effect = _httpx.ConnectError("refused")

            assert await client.health_check() is False


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfigFromEnv:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("MODEL_SERVER_URL", raising=False)
        monkeypatch.delenv("MODEL_SERVER_TIMEOUT", raising=False)
        monkeypatch.delenv("MODEL_SERVER_BATCH_SIZE", raising=False)
        c = ModelClient()
        assert c.base_url == "http://localhost:8001"
        assert c.timeout == 30.0
        assert c.batch_size == 32

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("MODEL_SERVER_URL", "http://prod-model:9000")
        monkeypatch.setenv("MODEL_SERVER_TIMEOUT", "60")
        monkeypatch.setenv("MODEL_SERVER_BATCH_SIZE", "64")
        c = ModelClient()
        assert c.base_url == "http://prod-model:9000"
        assert c.timeout == 60.0
        assert c.batch_size == 64

    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MODEL_SERVER_URL", "http://env-server")
        c = ModelClient(model_server_url="http://explicit-server")
        assert c.base_url == "http://explicit-server"

    def test_trailing_slash_stripped(self):
        c = ModelClient(model_server_url="http://server:8001/")
        assert c.base_url == "http://server:8001"
