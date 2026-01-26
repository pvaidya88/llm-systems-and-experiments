import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

Message = Dict[str, str]


class LLMClient:
    def complete(self, messages: List[Message]) -> str:
        raise NotImplementedError


class CallableLLM(LLMClient):
    def __init__(self, fn: Callable[[List[Message]], str]):
        self._fn = fn

    def complete(self, messages: List[Message]) -> str:
        return self._fn(messages)


@dataclass
class OpenAICompatibleClient(LLMClient):
    base_url: str
    api_key: Optional[str]
    model: str
    temperature: float = 0.2
    timeout: int = 60

    def complete(self, messages: List[Message]) -> str:
        if not self.base_url:
            raise ValueError("base_url is required")
        if not self.model:
            raise ValueError("model is required")

        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8")
            except Exception:
                err_body = "<no body>"
            raise RuntimeError(f"LLM request failed: {exc.code} {exc.reason}: {err_body}")

        parsed = json.loads(body)
        choices = parsed.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM response missing choices: {body}")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            content = choices[0].get("text")
        if content is None:
            raise RuntimeError(f"LLM response missing content: {body}")
        return content


@dataclass
class OpenAIResponsesClient(LLMClient):
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    reasoning_effort: Optional[str] = None
    text_verbosity: Optional[str] = "medium"
    text_format_type: Optional[str] = "text"
    store: Optional[bool] = None
    include: Optional[List[str]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    timeout: int = 60

    def __post_init__(self) -> None:
        self._client = None

    def complete(self, messages: List[Message]) -> str:
        if not self.model:
            raise ValueError("model is required")

        client = self._get_client()
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": _normalize_messages(messages),
        }

        text_config: Dict[str, Any] = {}
        if self.text_format_type:
            text_config["format"] = {"type": self.text_format_type}
        if self.text_verbosity:
            text_config["verbosity"] = self.text_verbosity
        if text_config:
            payload["text"] = text_config

        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.tools is not None:
            payload["tools"] = self.tools
        if self.store is not None:
            payload["store"] = self.store
        if self.include:
            payload["include"] = self.include

        response = client.responses.create(**payload)
        content = _extract_response_text(response)
        if not content:
            raise RuntimeError(f"LLM response missing text: {response}")
        return content

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError(
                "OpenAI SDK not installed. Install with `pip install openai` to use "
                "OpenAIResponsesClient."
            ) from exc

        kwargs: Dict[str, Any] = {}
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client


def _normalize_messages(messages: List[Message]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role is None or content is None:
            raise ValueError(f"Message missing role/content: {msg}")
        normalized.append({"role": str(role), "content": str(content)})
    return normalized


def _extract_response_text(response: Any) -> Optional[str]:
    if isinstance(response, dict):
        if response.get("output_text"):
            return response["output_text"]
        output = response.get("output") or []
    else:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text
        output = getattr(response, "output", None) or []

    chunks: List[str] = []
    for item in output:
        if isinstance(item, dict):
            item_text = item.get("text")
            content = item.get("content")
        else:
            item_text = getattr(item, "text", None)
            content = getattr(item, "content", None)

        if item_text:
            chunks.append(item_text)
        if not content:
            continue

        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
                part_text = part.get("text")
            else:
                part_type = getattr(part, "type", None)
                part_text = getattr(part, "text", None)
            if part_type in ("output_text", "text") and part_text:
                chunks.append(part_text)

    if chunks:
        return "".join(chunks)
    return None
