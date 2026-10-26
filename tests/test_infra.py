"""
tests/test_infra.py

Tests for infrastructure additions:
  - Per-IP rate limiter: unit tests of _RateLimiter and HTTP 429 from /inspect.
  - S3 persistence: boto3 failure must NEVER change the verdict.
"""

import json
import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from src.api.main import _RateLimiter, _upload_to_s3, app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png(width: int = 200, height: int = 200) -> bytes:
    """Return a minimal valid PNG of a grey image with a thin dark line."""
    img = np.full((height, width, 3), 140, dtype=np.uint8)
    img[height // 2, :, :] = 30
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Unit tests: _RateLimiter
# ---------------------------------------------------------------------------

class TestRateLimiterUnit:
    """Unit tests for the sliding-window _RateLimiter class."""

    def test_allows_requests_within_limit(self):
        """Up to max_per_minute requests from the same IP are allowed."""
        limiter = _RateLimiter(max_per_minute=5)
        for _ in range(5):
            assert limiter.is_allowed("1.2.3.4") is True

    def test_blocks_after_limit_exceeded(self):
        """The (max+1)th request in the same minute is denied."""
        limiter = _RateLimiter(max_per_minute=3)
        for _ in range(3):
            limiter.is_allowed("10.0.0.1")
        assert limiter.is_allowed("10.0.0.1") is False

    def test_different_ips_independent(self):
        """Each client IP has its own independent sliding window."""
        limiter = _RateLimiter(max_per_minute=2)
        limiter.is_allowed("192.168.1.1")
        limiter.is_allowed("192.168.1.1")
        assert limiter.is_allowed("192.168.1.1") is False
        assert limiter.is_allowed("192.168.1.2") is True

    def test_timestamps_expire_after_60_seconds(self):
        """Timestamps older than 60 s are evicted, restoring quota."""
        limiter = _RateLimiter(max_per_minute=2)
        ip = "172.16.0.5"
        limiter.is_allowed(ip)
        limiter.is_allowed(ip)
        assert limiter.is_allowed(ip) is False

        # Age existing timestamps beyond the 60-second window
        dq = limiter._windows[ip]
        old_times = list(dq)
        dq.clear()
        for t in old_times:
            dq.append(t - 61)

        # After eviction the slot is free again
        assert limiter.is_allowed(ip) is True

    def test_zero_limit_blocks_all(self):
        """max_per_minute=0 blocks every request immediately."""
        limiter = _RateLimiter(max_per_minute=0)
        assert limiter.is_allowed("any.ip") is False


# ---------------------------------------------------------------------------
# Integration tests: rate limit via HTTP 429
# ---------------------------------------------------------------------------

class TestRateLimitHTTP:
    """Integration tests: /inspect returns 429 when the limit is exhausted."""

    def test_exceeding_rate_limit_returns_429(self, monkeypatch):
        """Patch _rate_limiter.is_allowed to False -> expect HTTP 429."""
        from src.api import main as api_module

        mock_limiter = MagicMock()
        mock_limiter.is_allowed.return_value = False
        monkeypatch.setattr(api_module, "_rate_limiter", mock_limiter)

        response = client.post(
            "/inspect",
            files={"file": ("test.png", _make_png(), "image/png")},
        )
        assert response.status_code == 429
        body = response.json()
        # FastAPI wraps HTTPException with "detail"
        assert "detail" in body

    def test_within_rate_limit_returns_200(self, monkeypatch):
        """When is_allowed returns True, /inspect must return 200."""
        from src.api import main as api_module

        mock_limiter = MagicMock()
        mock_limiter.is_allowed.return_value = True
        monkeypatch.setattr(api_module, "_rate_limiter", mock_limiter)

        response = client.post(
            "/inspect",
            files={"file": ("test.png", _make_png(), "image/png")},
        )
        assert response.status_code == 200

    def test_health_endpoint_not_rate_limited(self):
        """GET /health has no rate-limit dependency and always returns 200."""
        response = client.get("/health")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Tests: S3 failure must never change the verdict
# ---------------------------------------------------------------------------

class TestS3Persistence:
    """S3 upload failures are non-fatal and must not affect the verdict."""

    def test_s3_failure_does_not_change_verdict(self, monkeypatch):
        """boto3.client raises -> /inspect still returns 200 with a valid verdict."""
        from src.api import main as api_module

        monkeypatch.setattr(api_module, "S3_BUCKET", "fake-bucket")
        # Prevent rate-limiter from blocking this test (quota may be consumed by earlier tests)
        mock_limiter = MagicMock()
        mock_limiter.is_allowed.return_value = True
        monkeypatch.setattr(api_module, "_rate_limiter", mock_limiter)

        import boto3
        with patch.object(boto3, "client", side_effect=Exception("Simulated S3 auth failure")):
            response = client.post(
                "/inspect",
                files={"file": ("crack_test.png", _make_png(), "image/png")},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["verdict"] in ("defect_found", "no_defect", "recapture_requested", "human_review")

    def test_s3_failure_payload_has_required_fields(self, monkeypatch):
        """Response has verdict, confidence, trace even when S3 is broken."""
        from src.api import main as api_module

        monkeypatch.setattr(api_module, "S3_BUCKET", "fake-bucket")
        mock_limiter = MagicMock()
        mock_limiter.is_allowed.return_value = True
        monkeypatch.setattr(api_module, "_rate_limiter", mock_limiter)

        import boto3
        with patch.object(boto3, "client", side_effect=RuntimeError("Simulated S3 timeout")):
            response = client.post(
                "/inspect",
                files={"file": ("any.png", _make_png(), "image/png")},
            )

        assert response.status_code == 200
        payload = response.json()
        assert "verdict" in payload
        assert "confidence" in payload
        assert "trace" in payload

    def test_s3_disabled_when_bucket_unset(self, monkeypatch):
        """When S3_BUCKET is None, boto3 must never be called."""
        from src.api import main as api_module

        monkeypatch.setattr(api_module, "S3_BUCKET", None)

        with patch("boto3.client") as mock_boto:
            _upload_to_s3("req-123", {"verdict": "no_defect"}, None)
            mock_boto.assert_not_called()

    def test_upload_to_s3_calls_put_object_with_correct_bucket(self, monkeypatch):
        """_upload_to_s3 calls s3.put_object with the configured bucket name."""
        from src.api import main as api_module

        monkeypatch.setattr(api_module, "S3_BUCKET", "my-test-bucket")

        mock_s3 = MagicMock()
        with patch("boto3.client", return_value=mock_s3):
            _upload_to_s3(
                "req-abc",
                {"verdict": "defect_found", "confidence": 0.9},
                None,
            )

        assert mock_s3.put_object.call_count >= 1
        first_kwargs = mock_s3.put_object.call_args_list[0][1]
        assert first_kwargs["Bucket"] == "my-test-bucket"
        assert first_kwargs["Key"].endswith("result.json")

    def test_upload_to_s3_never_logs_credentials(self, monkeypatch, caplog):
        """No credential-like strings appear in log output during upload."""
        import logging
        from src.api import main as api_module

        monkeypatch.setattr(api_module, "S3_BUCKET", "safe-bucket")
        mock_s3 = MagicMock()

        with patch("boto3.client", return_value=mock_s3):
            with caplog.at_level(logging.INFO, logger="sentry.api"):
                _upload_to_s3("req-safe", {"verdict": "no_defect"}, None)

        log_text = " ".join(caplog.messages)
        # Ensure nothing resembling an access key or secret appears
        for forbidden in ("AKIA", "aws_secret", "password", "token="):
            assert forbidden not in log_text
