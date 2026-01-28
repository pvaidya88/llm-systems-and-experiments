import os
from multiprocessing import get_context
from typing import Any, Dict, Optional

from .llm import LLMClient, OpenAICompatibleClient, OpenAIResponsesClient


class NoteOracleError(RuntimeError):
    pass


def ask_yesno_with_client(
    client: LLMClient,
    prompt: str,
    max_retries: int,
    system: Optional[str] = "Answer yes or no only.",
) -> str:
    retries = max(1, max_retries)
    last_response = ""
    for attempt in range(retries):
        if attempt == 0:
            response = _complete(client, prompt, system=system)
        else:
            response = _complete(
                client,
                f"Answer yes or no only. {prompt}",
                system="Answer yes or no only.",
            )
        last_response = response or ""
        normalized = _normalize_yesno(last_response)
        if normalized is not None:
            return normalized
    raise NoteOracleError(
        f"note_yesno failed after {retries} retries: {last_response!r}"
    )


def _complete(client: LLMClient, prompt: str, system: Optional[str] = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": str(system)})
    messages.append({"role": "user", "content": str(prompt)})
    return client.complete(messages)


def _normalize_yesno(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    cleaned = text.strip().lower()
    if cleaned.startswith("y"):
        return "yes"
    if cleaned.startswith("n"):
        return "no"
    return None


class NoteOracleClient:
    def __init__(self, conn, process):
        self._conn = conn
        self._process = process
        self._closed = False

    @classmethod
    def from_llm(cls, llm: LLMClient, note: str, max_retries: Optional[int] = None):
        spec = _serialize_llm_client(llm)
        if spec is None:
            raise NoteOracleError(
                "Unsupported sub-LLM type for isolated note oracle."
            )
        retries = max_retries
        if retries is None:
            retries = max(1, int(os.environ.get("LLM_YESNO_MAX_RETRIES", "4")))
        ctx = get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        process = ctx.Process(
            target=_note_oracle_server,
            args=(child_conn, note, spec, int(retries)),
            daemon=True,
        )
        process.start()
        if not parent_conn.poll(10.0):
            process.terminate()
            raise NoteOracleError("note oracle startup timed out")
        ready = parent_conn.recv()
        if not isinstance(ready, dict) or not ready.get("ok"):
            error = "note oracle failed to start"
            if isinstance(ready, dict) and ready.get("error"):
                error = ready["error"]
            process.join(timeout=0.5)
            raise NoteOracleError(error)
        return cls(parent_conn, process)

    def note_yesno(self, question: str) -> str:
        if self._closed:
            raise NoteOracleError("note oracle is closed")
        self._conn.send({"type": "note_yesno", "question": str(question)})
        response = self._conn.recv()
        if not isinstance(response, dict) or not response.get("ok"):
            error = "note oracle error"
            if isinstance(response, dict) and response.get("error"):
                error = response["error"]
            raise NoteOracleError(error)
        return str(response.get("answer", ""))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._conn.send({"type": "close"})
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass
        if self._process.is_alive():
            self._process.join(timeout=0.5)
        if self._process.is_alive():
            self._process.terminate()


def _serialize_llm_client(llm: LLMClient) -> Optional[Dict[str, Any]]:
    if isinstance(llm, OpenAICompatibleClient):
        return {
            "kind": "openai_compatible",
            "base_url": llm.base_url,
            "api_key": llm.api_key,
            "model": llm.model,
            "temperature": llm.temperature,
            "timeout": llm.timeout,
        }
    if isinstance(llm, OpenAIResponsesClient):
        return {
            "kind": "openai_responses",
            "model": llm.model,
            "api_key": llm.api_key,
            "base_url": llm.base_url,
            "reasoning_effort": llm.reasoning_effort,
            "text_verbosity": llm.text_verbosity,
            "text_format_type": llm.text_format_type,
            "store": llm.store,
            "include": llm.include,
            "tools": llm.tools,
            "timeout": llm.timeout,
        }
    return None


def _deserialize_llm_client(spec: Dict[str, Any]) -> LLMClient:
    kind = spec.get("kind")
    if kind == "openai_compatible":
        return OpenAICompatibleClient(
            base_url=spec.get("base_url") or "",
            api_key=spec.get("api_key"),
            model=spec.get("model") or "",
            temperature=spec.get("temperature", 0.2),
            timeout=spec.get("timeout", 60),
        )
    if kind == "openai_responses":
        return OpenAIResponsesClient(
            model=spec.get("model") or "",
            api_key=spec.get("api_key"),
            base_url=spec.get("base_url"),
            reasoning_effort=spec.get("reasoning_effort"),
            text_verbosity=spec.get("text_verbosity"),
            text_format_type=spec.get("text_format_type"),
            store=spec.get("store"),
            include=spec.get("include"),
            tools=spec.get("tools"),
            timeout=spec.get("timeout", 60),
        )
    raise NoteOracleError(f"Unknown LLM spec: {kind}")


def _note_oracle_server(conn, note: str, spec: Dict[str, Any], max_retries: int) -> None:
    try:
        client = _deserialize_llm_client(spec)
    except Exception as exc:
        conn.send({"ok": False, "error": str(exc)})
        return

    conn.send({"ok": True})
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if not isinstance(msg, dict):
            continue
        msg_type = msg.get("type")
        if msg_type == "close":
            break
        if msg_type != "note_yesno":
            conn.send({"ok": False, "error": "unsupported message"})
            continue
        question = str(msg.get("question", ""))
        prompt = f"{question} Note: {note}"
        try:
            answer = ask_yesno_with_client(
                client, prompt, max_retries, system="Answer yes or no only."
            )
            conn.send({"ok": True, "answer": answer})
        except Exception as exc:
            conn.send({"ok": False, "error": str(exc)})
    try:
        conn.close()
    except Exception:
        pass
