import os
import time
from multiprocessing import get_context
from typing import Any, Callable, Dict, Optional

from .repl_worker import worker_main


class ReplEnvironment:
    def __init__(
        self,
        context: Any,
        llm_query: Callable,
        max_output_chars: int = 4000,
        note_yesno: Optional[Callable[[str], str]] = None,
        rlm_query: Optional[Callable[[str, Any, int], str]] = None,
        trace_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
        repl_timeout_s: float = 5.0,
        repl_memory_mb: Optional[int] = 256,
        repl_cpu_seconds: Optional[int] = 5,
        enable_ctx: bool = True,
        max_snippet_chars: int = 200,
        cache_enabled: bool = True,
        tool_registry: Optional[Any] = None,
        remote_tools_enabled: bool = False,
    ):
        self._llm_query = llm_query
        self._note_yesno = note_yesno
        self._rlm_query = rlm_query
        self._trace_hook = trace_hook
        self._tool_registry = tool_registry
        self._timeout_s = float(repl_timeout_s)
        self._yesno_max_retries = max(
            1, int(os.environ.get("LLM_YESNO_MAX_RETRIES", "4"))
        )

        ctx = get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        config = {
            "max_output_chars": int(max_output_chars),
            "enable_note_yesno": note_yesno is not None,
            "enable_rlm_query": rlm_query is not None,
            "enable_ctx": bool(enable_ctx),
            "max_snippet_chars": int(max_snippet_chars),
            "memory_mb": repl_memory_mb,
            "cpu_seconds": repl_cpu_seconds,
            "cache_enabled": bool(cache_enabled),
            "enable_tools": bool(remote_tools_enabled and tool_registry is not None),
        }
        self._process = ctx.Process(
            target=worker_main,
            args=(child_conn, context, config),
            daemon=True,
        )
        self._process.start()
        self._conn = parent_conn

    def exec(self, code: str) -> Dict[str, Any]:
        self._conn.send({"type": "exec", "code": code})
        deadline = time.monotonic() + self._timeout_s
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0:
                self._terminate("REPL timeout exceeded")
            if not self._conn.poll(remaining):
                continue
            try:
                msg = self._conn.recv()
            except EOFError:
                self._terminate("REPL process terminated")
            if not isinstance(msg, dict):
                continue
            msg_type = msg.get("type")
            if msg_type == "exec_result":
                return {
                    "output": str(msg.get("output") or ""),
                    "truncated": bool(msg.get("truncated")),
                    "error": msg.get("error"),
                }
            self._handle_rpc(msg)

    def get_var(self, name: str) -> Any:
        self._conn.send({"type": "get_var", "name": name})
        deadline = time.monotonic() + self._timeout_s
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0:
                self._terminate("REPL timeout exceeded")
            if not self._conn.poll(remaining):
                continue
            try:
                msg = self._conn.recv()
            except EOFError:
                self._terminate("REPL process terminated")
            if not isinstance(msg, dict):
                continue
            msg_type = msg.get("type")
            if msg_type == "get_var_result":
                if msg.get("error"):
                    raise KeyError(str(msg.get("error")))
                return msg.get("content")
            self._handle_rpc(msg)

    def close(self) -> None:
        if self._process is None:
            return
        try:
            self._conn.send({"type": "shutdown"})
        except Exception:
            pass
        self._process.join(timeout=0.5)
        if self._process.is_alive():
            self._process.terminate()
        self._process = None

    def _terminate(self, reason: str) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
        raise RuntimeError(reason)

    def _handle_rpc(self, msg: Dict[str, Any]) -> None:
        msg_type = msg.get("type")
        if msg_type == "llm_query":
            try:
                result = self._call_llm_query(msg.get("prompt"), msg.get("system"))
                self._respond("llm_query_result", result)
            except Exception as exc:
                self._respond_error("llm_query_result", str(exc))
            return
        if msg_type == "llm_query_yesno":
            try:
                result = self._call_llm_query_yesno(
                    msg.get("prompt"),
                    msg.get("system"),
                    msg.get("max_retries"),
                )
                self._respond("llm_query_yesno_result", result)
            except Exception as exc:
                self._respond_error("llm_query_yesno_result", str(exc))
            return
        if msg_type == "note_yesno":
            if self._note_yesno is None:
                self._respond_error("note_yesno_result", "note_yesno is unavailable")
            else:
                try:
                    self._respond(
                        "note_yesno_result", self._note_yesno(str(msg.get("question")))
                    )
                except Exception as exc:
                    self._respond_error("note_yesno_result", str(exc))
            return
        if msg_type == "rlm_query":
            if self._rlm_query is None:
                self._respond_error("rlm_query_result", "rlm_query is unavailable")
            else:
                try:
                    question = str(msg.get("question"))
                    sub_context = msg.get("sub_context")
                    depth_limit = int(msg.get("depth_limit", 1))
                    result = self._rlm_query(question, sub_context, depth_limit)
                    self._respond("rlm_query_result", result)
                except Exception as exc:
                    self._respond_error("rlm_query_result", str(exc))
            return
        if msg_type == "ctx_event":
            if self._trace_hook is not None:
                event = msg.get("event")
                if isinstance(event, dict):
                    self._trace_hook(event)
            self._respond("ctx_event_ack", "")
        if msg_type == "tool_call":
            if self._tool_registry is None:
                self._respond_error("tool_result", "tool_registry is unavailable")
            else:
                try:
                    name = str(msg.get("name"))
                    args = msg.get("args") or []
                    kwargs = msg.get("kwargs") or {}
                    result = self._tool_registry.call(name, args, kwargs)
                    self._respond("tool_result", result)
                except Exception as exc:
                    self._respond_error("tool_result", str(exc))
            return

    def _call_llm_query(self, prompt: Any, system: Optional[str]) -> str:
        return self._llm_query(prompt, system)

    def _call_llm_query_yesno(
        self, prompt: Any, system: Optional[str], max_retries: Optional[int]
    ) -> str:
        retries = max(1, max_retries or self._yesno_max_retries)
        last_response = ""
        for attempt in range(retries):
            if attempt == 0:
                response = self._call_llm_query(prompt, system)
            else:
                response = self._call_llm_query(
                    f"Answer yes or no only. {prompt}",
                    "Answer yes or no only.",
                )
            last_response = response or ""
            normalized = self._normalize_yesno(last_response)
            if normalized is not None:
                return normalized
        raise RuntimeError(
            f"llm_query_yesno failed after {retries} retries: {last_response!r}"
        )

    @staticmethod
    def _normalize_yesno(text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        cleaned = text.strip().lower()
        if cleaned.startswith("y"):
            return "yes"
        if cleaned.startswith("n"):
            return "no"
        return None

    def _respond(self, msg_type: str, content: Any) -> None:
        self._conn.send({"type": msg_type, "content": content})

    def _respond_error(self, msg_type: str, error: str) -> None:
        self._conn.send({"type": msg_type, "error": error})

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
