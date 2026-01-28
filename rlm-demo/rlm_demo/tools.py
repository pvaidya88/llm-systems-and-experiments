import json
from typing import Any, Callable, Dict, Optional


def _is_json_safe(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except Exception:
        return False


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


class ToolRegistry:
    def __init__(self, max_result_chars: int = 2000):
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._max_result_chars = max_result_chars

    def register(self, name: str, func: Callable[..., Any]) -> None:
        self._tools[name] = func

    def call(self, name: str, args: list, kwargs: dict) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool not registered: {name}")
        result = self._tools[name](*args, **kwargs)
        return self._sanitize(result)

    def _sanitize(self, result: Any) -> Any:
        if isinstance(result, str):
            return _truncate_text(result, self._max_result_chars)
        if isinstance(result, list):
            trimmed = []
            for item in result:
                trimmed.append(self._sanitize(item))
            return trimmed
        if isinstance(result, dict):
            trimmed = {}
            for key, value in result.items():
                if isinstance(value, str):
                    trimmed[key] = _truncate_text(value, self._max_result_chars)
                else:
                    trimmed[key] = self._sanitize(value)
            return trimmed
        if not _is_json_safe(result):
            return str(result)
        return result


class ToolProxy:
    def __init__(self, call_fn: Callable[[str, list, dict], Any]):
        self._call_fn = call_fn

    def __getattr__(self, name: str) -> Callable[..., Any]:
        def _wrapped(*args, **kwargs):
            return self._call_fn(name, list(args), kwargs)
        return _wrapped
