from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request


class OllamaError(RuntimeError):
    pass


@dataclass
class OllamaClient:
    base_url: str = "http://localhost:11434"
    model_name: str = "llama3.1"
    temperature: float = 0.2
    timeout: int = 120

    def chat_raw(self, messages: list[dict[str, Any]], stream: bool = False) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "format": "json",
            "options": {"temperature": self.temperature},
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except error.URLError as exc:
            raise OllamaError(
                f"Could not connect to Ollama at {self.base_url}. Make sure Ollama is running."
            ) from exc
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            raise OllamaError(f"Ollama chat request failed: {exc}") from exc

        if stream:
            return body

        response = json.loads(body)
        message = response.get("message", {})
        content = message.get("content", "")
        if not content:
            raise OllamaError("Ollama response did not contain message.content")
        return content

    def chat(self, messages: list[dict[str, Any]], stream: bool = False) -> str:
        return self.chat_raw(messages=messages, stream=stream)
