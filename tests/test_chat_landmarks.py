"""Chat service answers landmark-case prompts (Judgments tab → Ask in Assistant)."""

from backend.services.chat_service import generate_chat_response


def test_bachan_singh_bullet_prompt_gets_landmark_briefing():
    msg = (
        "In 5–8 bullet points, explain Bachan Singh v. State of Punjab "
        "(1980 SCC (2) 684) and what practitioners typically take from it."
    )
    out = generate_chat_response(msg, session_id=None)
    text = out["response"]
    assert "Bachan Singh" in text
    assert "rarest" in text.lower()
    assert "document-analysis mode" not in text.lower()


def test_arnesh_kumar_matched():
    out = generate_chat_response("What did Arnesh Kumar decide?", session_id=None)
    assert "Arnesh Kumar" in out["response"]
    assert "bail" in out["response"].lower() or "arrest" in out["response"].lower()
