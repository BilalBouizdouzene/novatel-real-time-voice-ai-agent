"""Asynchronous streaming client for any OpenAI-compatible chat endpoint."""

import json
import logging
from collections.abc import AsyncIterator

import httpx

from app.llm.prompts import build_messages
from app.models.schemas import RetrievedDocument

logger = logging.getLogger(__name__)


class LLMFailure(RuntimeError):
    """Raised for connection, protocol or provider errors."""


class OpenAICompatibleLLMClient:
    """Stream assistant text from Gemini, Ollama, or another compatible provider."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_seconds: float = 90.0,
    ) -> None:
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    async def stream_chat(
        self,
        user_query: str,
        retrieved_context: list[RetrievedDocument],
        conversation_history: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        """Yield content deltas; credentials are optional for local Ollama."""

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": build_messages(user_query, retrieved_context, conversation_history),
            "stream": True,
            "temperature": 0.2,
        }
        try:
            timeout = httpx.Timeout(self.timeout_seconds, connect=min(self.timeout_seconds, 10.0))
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", self.endpoint, headers=headers, json=payload) as response:
                    if response.is_error:
                        logger.warning("LLM server returned HTTP %s", response.status_code)
                        raise LLMFailure(f"LLM server returned HTTP {response.status_code}")
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            logger.warning("Ignoring malformed LLM stream event")
                            continue
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            yield content
        except LLMFailure:
            raise
        except (httpx.HTTPError, OSError) as exc:
            logger.exception("LLM provider is unreachable")
            raise LLMFailure("LLM provider is unreachable; verify LLM_BASE_URL and provider availability") from exc
        except Exception as exc:
            logger.exception("LLM response stream failed")
            raise LLMFailure("LLM response could not be read") from exc
