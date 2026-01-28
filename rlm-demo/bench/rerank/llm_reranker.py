import hashlib
import json
import os
from typing import List, Dict, Any, Optional

from ..llm_clients import LLMWrapper
from rlm_demo.llm import OpenAIResponsesClient


class LLMReranker:
    def __init__(self, model: Optional[str] = None, cache_path: Optional[str] = None):
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5.2")
        self.cache_path = cache_path
        self._cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as handle:
                try:
                    self._cache = json.load(handle)
                except Exception:
                    self._cache = {}

        client = OpenAIResponsesClient(
            model=self.model,
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            reasoning_effort=os.environ.get("OPENAI_REASONING_EFFORT"),
            text_verbosity="low",
        )
        self._llm = LLMWrapper(client)

    @property
    def stats(self):
        return self._llm.stats

    def _cache_key(self, question: str, candidates: List[Dict[str, Any]]) -> str:
        ids = ",".join(str(c.get("chunk_id")) for c in candidates)
        data = f"{question}|{ids}".encode("utf-8")
        return hashlib.sha1(data).hexdigest()

    def rerank(self, question: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        cache_key = self._cache_key(question, candidates)
        if cache_key in self._cache:
            ranked_ids = self._cache[cache_key]
        else:
            prompt = self._build_prompt(question, candidates, top_k)
            response = self._llm.complete([
                {"role": "system", "content": "You are a strict reranker."},
                {"role": "user", "content": prompt},
            ])
            ranked_ids = self._parse_ids(response)
            self._cache[cache_key] = ranked_ids
            if self.cache_path:
                with open(self.cache_path, "w", encoding="utf-8") as handle:
                    json.dump(self._cache, handle, ensure_ascii=False, indent=2)

        ranked = []
        id_to_hit = {str(c.get("chunk_id")): c for c in candidates}
        for cid in ranked_ids:
            if cid in id_to_hit:
                ranked.append(id_to_hit[cid])
        if len(ranked) < top_k:
            for hit in candidates:
                if hit not in ranked:
                    ranked.append(hit)
                if len(ranked) >= top_k:
                    break
        return ranked[:top_k]

    def _build_prompt(self, question: str, candidates: List[Dict[str, Any]], top_k: int) -> str:
        lines = [
            "Rank the candidate passages for answering the question.",
            "Return JSON with key 'ranked_ids' as a list of chunk_id strings.",
            f"Return exactly {top_k} ids.",
            f"Question: {question}",
            "Candidates:",
        ]
        for c in candidates:
            text = str(c.get("text", ""))[:400]
            lines.append(f"- chunk_id={c.get('chunk_id')} doc_id={c.get('doc_id')}: {text}")
        return "\n".join(lines)

    def _parse_ids(self, text: str) -> List[str]:
        try:
            data = json.loads(text)
            ids = data.get("ranked_ids") or []
            return [str(x) for x in ids]
        except Exception:
            # fallback: extract tokens between brackets
            ids = []
            for token in text.replace("[", " ").replace("]", " ").replace(",", " ").split():
                if token.isdigit() or token.startswith("chunk"):
                    ids.append(token.strip())
            return ids
