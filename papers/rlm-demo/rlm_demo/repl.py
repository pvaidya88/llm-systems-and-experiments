import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict


class ReplEnvironment:
    def __init__(self, context: Any, llm_query: Callable, max_output_chars: int = 4000):
        self._max_output_chars = max_output_chars
        self._globals: Dict[str, Any] = {
            "context": context,
            "llm_query": llm_query,
        }
        self._locals: Dict[str, Any] = {}

    def exec(self, code: str) -> str:
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(code, self._globals, self._locals)
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