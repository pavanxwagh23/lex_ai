"""
backend/ai_client/llm_client.py
================================
Abstraction layer for Large Language Model (LLM) interactions.

Provides a unified ``LLMClient`` class that can be swapped between:

- **Mock mode** — instant canned responses (default, no API key needed)
- **OpenAI mode** — calls OpenAI API if ``OPENAI_API_KEY`` is set

This allows the chat service to generate natural language responses for
general legal questions without coupling to a specific LLM provider.
"""

from __future__ import annotations

import os
from typing import Optional

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Thin wrapper around an LLM provider.

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key.  Falls back to ``os.environ["OPENAI_API_KEY"]``.
        If neither is available, the client operates in **mock mode**.
    model : str
        OpenAI model identifier (default ``"gpt-3.5-turbo"``).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
    ) -> None:
        self._api_key: Optional[str] = api_key or os.getenv("OPENAI_API_KEY")
        self._model: str = model
        self._client = None  # lazy-loaded OpenAI client

        if self._api_key:
            logger.info("LLMClient initialised in LIVE mode (model=%s).", model)
        else:
            logger.info(
                "LLMClient initialised in MOCK mode — "
                "set OPENAI_API_KEY for real LLM responses."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a text response from the LLM.

        Parameters
        ----------
        prompt : str
            The user-facing prompt / question.
        system_prompt : str, optional
            System-level instruction (e.g. "You are a legal AI assistant").

        Returns
        -------
        str
            LLM response text.
        """
        if not self._api_key:
            return self._mock_response(prompt)

        return self._openai_response(prompt, system_prompt)

    # ------------------------------------------------------------------
    # OpenAI integration
    # ------------------------------------------------------------------

    def _openai_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call the OpenAI ChatCompletion API."""
        try:
            if self._client is None:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)

            messages = []
            if system_prompt:
                enhanced_system_prompt = (
                    f"{system_prompt}\n\n"
                    "CRITICAL INSTRUCTIONS:\n"
                    "- Explain legal concepts in simple, plain English.\n"
                    "- Avoid complex legal jargon.\n"
                    "- Keep answers concise and precise (max 5 lines where possible).\n"
                    "- Output directly; do not hallucinate JSON wrappers unless explicitly requested."
                )
                messages.append({"role": "system", "content": enhanced_system_prompt})
            else:
                messages.append({
                    "role": "system", 
                    "content": "You are a precise, deterministic legal assistant. Keep answers concise, avoid jargon, max 5 lines."
                })
            
            messages.append({"role": "user", "content": prompt})

            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=1024,
                temperature=0.0,  # Enforce deterministic outputs
            )

            result = response.choices[0].message.content.strip()
            logger.info("LLM response generated (%d chars).", len(result))
            return result

        except Exception as exc:
            logger.warning("OpenAI call failed (%s), falling back to mock.", exc)
            return self._mock_response(prompt)

    # ------------------------------------------------------------------
    # Mock fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_response(prompt: str) -> str:
        """Return a helpful mock response when no LLM is available."""
        prompt_lower = prompt.lower()

        if any(kw in prompt_lower for kw in ("hello", "hi", "hey", "greet")):
            return (
                "Hello! I'm Lex AI, your legal document assistant. "
                "I can summarize contracts, detect risks, compare documents, "
                "and answer general legal questions. How can I help you today?"
            )

        if any(kw in prompt_lower for kw in ("help", "what can you do", "capabilities")):
            return (
                "I can help you with the following:\n\n"
                "• **Summarize** — Get a structured summary of your contract\n"
                "• **Risk Analysis** — Identify legal risks and red flags\n"
                "• **Compare** — Compare two contracts side by side\n"
                "• **General Questions** — Ask me anything about legal terms\n\n"
                "Try saying: 'Summarize this contract' or 'What are the risks?'"
            )

        return (
            "I'm Lex AI, your legal document assistant. "
            "I can summarize contracts, analyze risks, and compare documents. "
            "Try asking me to 'summarize this contract' or 'analyze risks'. "
            "For full LLM-powered responses, set the OPENAI_API_KEY environment variable."
        )


# ---------------------------------------------------------------------------
# Module-level singleton (lazy-loaded)
# ---------------------------------------------------------------------------
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Return the global LLMClient singleton.

    Creates the instance on first call and reuses it thereafter.
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
