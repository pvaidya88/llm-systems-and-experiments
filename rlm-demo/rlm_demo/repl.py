import io
import os
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, Optional


class ReplEnvironment:
    def __init__(self, context: Any, llm_query: Callable, max_output_chars: int = 4000):
        self._max_output_chars = max_output_chars
        self._llm_query = llm_query
        self._yesno_max_retries = max(
            1, int(os.environ.get("LLM_YESNO_MAX_RETRIES", "4"))
        )
        self._globals: Dict[str, Any] = {
            "context": context,
            "llm_query": llm_query,
            "llm_query_yesno": self._llm_query_yesno,
        }
        self._locals: Dict[str, Any] = {}

    def _llm_query_yesno(
        self, prompt: str, system: Optional[str] = None, max_retries: Optional[int] = None
    ) -> str:
        retries = max(1, max_retries or self._yesno_max_retries)
        last_response = ""
        for attempt in range(retries):
            if attempt == 0:
                response = self._llm_query(prompt, system=system)
            else:
                response = self._llm_query(
                    f"Answer yes or no only. {prompt}",
                    system="Answer yes or no only.",
                )
            last_response = response or ""
            normalized = self._normalize_yesno(last_response)
            if normalized is not None:
                return normalized
        raise RuntimeError(f"llm_query_yesno failed after {retries} retries: {last_response!r}")

    @staticmethod
    def _normalize_yesno(text: str) -> Optional[str]:
        if text is None:
            return None
        cleaned = text.strip().lower()
        if cleaned.startswith("y"):
            return "yes"
        if cleaned.startswith("n"):
            return "no"
        return None

    def exec(self, code: str) -> str:
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(code, self._globals)
        except Exception:
            err = traceback.format_exc()
            combined = (stdout.getvalue() + stderr.getvalue() + err).strip()
            return self._truncate(combined)

        combined = (stdout.getvalue() + stderr.getvalue()).strip()
        if not combined:
            combined = "<no output>"
        return self._truncate(combined)

    def get_var(self, name: str) -> Any:
        if name in self._locals:
            return self._locals[name]
        if name in self._globals:
            return self._globals[name]
        raise KeyError(f"Variable not found: {name}")

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_output_chars:
            return text
        return text[: self._max_output_chars] + "\n...<truncated>"
