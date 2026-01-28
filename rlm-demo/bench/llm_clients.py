import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMCallStats:
    calls: int = 0
    total_wall_s: float = 0.0
    total_prompt_chars: int = 0
    total_output_chars: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def record(self, prompt_chars: int, output_chars: int, wall_s: float, usage: Optional[Dict[str, Any]] = None) -> None:
        self.calls += 1
        self.total_wall_s += wall_s
        self.total_prompt_chars += prompt_chars
        self.total_output_chars += output_chars
        if usage:
            for key, val in usage.items():
                self.extra[key] = self.extra.get(key, 0) + (val or 0)


class EmbeddingClient:
    dim: int = 0

    def embed_texts(self, texts: List[str]):
        raise NotImplementedError

    def embed_query(self, text: str):
        raise NotImplementedError


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.stats = LLMCallStats()
        self._client = None
        self.dim = 0

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            kwargs: Dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def embed_texts(self, texts: List[str]):
        if not texts:
            return []
        client = self._get_client()
        start = time.perf_counter()
        response = client.embeddings.create(model=self.model, input=texts)
        wall_s = time.perf_counter() - start
        vectors = [item.embedding for item in response.data]
        if vectors:
            self.dim = len(vectors[0])
        prompt_chars = sum(len(t) for t in texts)
        self.stats.record(prompt_chars, 0, wall_s, getattr(response, "usage", None))
        return vectors

    def embed_query(self, text: str):
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else []


class FakeEmbeddingClient(EmbeddingClient):
    def __init__(self, dim: int = 8):
        self.dim = dim

    def _hash_vec(self, text: str):
        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        return vec

    def embed_texts(self, texts: List[str]):
        return [self._hash_vec(text) for text in texts]

    def embed_query(self, text: str):
        return self._hash_vec(text)


class LLMWrapper:
    def __init__(self, llm_client, stats: Optional[LLMCallStats] = None):
        self._client = llm_client
        self.stats = stats or LLMCallStats()

    def complete(self, messages):
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        start = time.perf_counter()
        response = self._client.complete(messages)
        wall_s = time.perf_counter() - start
        output_chars = len(response or "")
        self.stats.record(prompt_chars, output_chars, wall_s)
        return response
