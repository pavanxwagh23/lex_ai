from types import SimpleNamespace

import pytest

from backend.services import rate_limiter
from backend.utils.error_handler import AppError


def _request(ip="127.0.0.1"):
    return SimpleNamespace(client=SimpleNamespace(host=ip))


def test_rate_limit_allows_requests_under_limit(monkeypatch):
    monkeypatch.setattr(rate_limiter, "_RATE_STORE", {})
    monkeypatch.setattr(rate_limiter, "RATE_LIMIT", 2)

    rate_limiter.rate_limit(_request())
    rate_limiter.rate_limit(_request())


def test_rate_limit_blocks_requests_over_limit(monkeypatch):
    monkeypatch.setattr(rate_limiter, "_RATE_STORE", {})
    monkeypatch.setattr(rate_limiter, "RATE_LIMIT", 1)

    rate_limiter.rate_limit(_request())

    with pytest.raises(AppError) as exc:
        rate_limiter.rate_limit(_request())

    assert exc.value.status_code == 429
