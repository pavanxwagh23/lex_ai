from backend.ai_client.llm_client import LocalHTTPResponder, RuleBasedResponder


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": " local answer "}}]}


def test_local_http_responder_posts_openai_compatible_payload(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse()

    monkeypatch.setattr("backend.ai_client.llm_client.requests.post", fake_post)

    responder = LocalHTTPResponder(base_url="http://localhost:11435/v1", model="lex-ai-legal", timeout=7)
    result = responder.generate("Explain warranty.", "You are careful.")

    assert result == "local answer"
    assert calls[0]["url"] == "http://localhost:11435/v1/chat/completions"
    assert calls[0]["json"]["model"] == "lex-ai-legal"
    assert calls[0]["json"]["messages"][0]["role"] == "system"
    assert calls[0]["json"]["messages"][1]["role"] == "user"
    assert calls[0]["timeout"] == 7


def test_rule_responder_matches_landmark_cases_like_chat_service():
    """RuleBasedResponder must use the same landmark briefings as /chat (summary LLM path)."""
    r = RuleBasedResponder()
    out = r.generate("Explain Bachan Singh v. State of Punjab briefly.", None)
    assert "Bachan Singh" in out
    assert "rarest" in out.lower()
    assert "offline, document-analysis mode" not in out.lower()
