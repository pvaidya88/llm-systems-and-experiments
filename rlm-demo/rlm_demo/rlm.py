import json
import os
import re
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .llm import LLMClient
from .note_oracle import NoteOracleClient, NoteOracleError
from .prompts import render_system_prompt
from .repl import ReplEnvironment
from .trace import RLMTrace, RetrievalEvent, RetrievalHit, ReplBlockTrace, SubcallTrace

RE_REPL_BLOCK = re.compile(r"```repl\s*(.*?)```", re.DOTALL | re.IGNORECASE)
RE_FINAL_VAR_LINE = re.compile(r"^\s*FINAL_VAR\(([^\n]*)\)\s*$")
RE_FINAL_LINE = re.compile(r"^\s*FINAL\(([^\n]*)\)\s*$")


@dataclass
class RLMOptions:
    max_steps: int = 8
    max_output_chars: int = 4000
    max_sub_calls: int = 32
    max_total_repl_output_chars: Optional[int] = None
    max_total_surfaced_chars: Optional[int] = None
    max_total_subcall_input_chars: Optional[int] = None
    repl_timeout_s: float = 5.0
    repl_memory_mb: Optional[int] = 256
    repl_cpu_seconds: Optional[int] = 5
    enable_ctx: bool = True
    max_trace_snippet_chars: int = 200
    protocol: str = "markdown"
    include_cost_hint: bool = True
    require_repl: bool = False
    retry_on_invalid: bool = False
    log_repl_outputs: bool = False
    min_sub_calls: int = 0
    min_rlm_queries: int = 0
    hidden_note: Optional[str] = None
    trace: Optional[RLMTrace] = None
    cache_enabled: bool = True


class RLM:
    def __init__(
        self,
        root_llm: LLMClient,
        sub_llm: Optional[LLMClient] = None,
        options: Optional[RLMOptions] = None,
        _depth: int = 0,
        _depth_limit: Optional[int] = None,
        _budget: Optional[Dict[str, int]] = None,
        _trace: Optional[RLMTrace] = None,
        _cache: Optional[Dict[str, str]] = None,
    ) -> None:
        self._root_llm = root_llm
        self._sub_llm = sub_llm or root_llm
        self._options = options or RLMOptions()
        self._depth = _depth
        self._depth_limit = _depth_limit
        self._budget = _budget
        self._trace = _trace or self._options.trace
        self._cache = _cache
        self._last_trace: Optional[RLMTrace] = None

    @property
    def last_trace(self) -> Optional[RLMTrace]:
        return self._last_trace

    def answer(self, query: str, context: Any) -> str:
        context_type, total_length, chunk_lengths = _describe_context(context)
        system_prompt = render_system_prompt(
            context_type=context_type,
            context_total_length=total_length,
            context_lengths=chunk_lengths,
            include_cost_hint=self._options.include_cost_hint,
            protocol=self._options.protocol,
        )

        trace = self._trace or RLMTrace()
        self._trace = trace
        self._last_trace = trace
        trace.termination_reason = None

        if self._options.max_total_repl_output_chars is None:
            max_total_repl_output = self._options.max_output_chars * max(
                1, self._options.max_steps
            )
        else:
            max_total_repl_output = self._options.max_total_repl_output_chars

        if self._options.max_total_surfaced_chars is None:
            max_total_surfaced = max_total_repl_output * 2
        else:
            max_total_surfaced = self._options.max_total_surfaced_chars

        if self._options.max_total_subcall_input_chars is None:
            max_total_subcall_input = max(1, self._options.max_sub_calls) * 4000
        else:
            max_total_subcall_input = self._options.max_total_subcall_input_chars

        budget = getattr(self, "_budget", None)
        if budget is None:
            budget = {
                "repl_output": 0,
                "surfaced": 0,
                "subcall_input": 0,
                "model_input": 0,
            }
            self._budget = budget

        if self._cache is None and self._options.cache_enabled:
            self._cache = {}

        subcall_state = {"count": 0}

        def _bump_subcall(prompt_chars: int, system_chars: int) -> None:
            subcall_state["count"] += 1
            if subcall_state["count"] > self._options.max_sub_calls:
                raise RuntimeError("llm_query call limit exceeded")
            budget["subcall_input"] += prompt_chars + system_chars
            trace.subcall_input_chars = budget["subcall_input"]
            if max_total_subcall_input is not None and budget["subcall_input"] > max_total_subcall_input:
                raise RuntimeError("Total subcall input budget exceeded")

        def llm_query(prompt: Any, system: Optional[str] = None) -> str:
            prompt_str = str(prompt)
            system_str = str(system) if system else ""
            _bump_subcall(len(prompt_str), len(system_str))
            messages: List[Dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system_str})
            messages.append({"role": "user", "content": prompt_str})
            response = self._sub_llm.complete(messages)
            trace.subcalls.append(
                SubcallTrace(
                    prompt_chars=len(prompt_str),
                    system_chars=len(system_str),
                    response_chars=len(response or ""),
                )
            )
            trace.subcall_output_chars += len(response or "")
            return response

        def trace_hook(event: Dict[str, Any]) -> None:
            cache_hit = None
            params_in = event.get("params") or {}
            if "cache_hit" in params_in:
                cache_hit = bool(params_in.get("cache_hit"))
                if cache_hit:
                    trace.retrieval_cache_hits += 1
                else:
                    trace.retrieval_cache_misses += 1
            hits = []
            for hit in event.get("hits", []):
                hits.append(
                    RetrievalHit(
                        ref_id=str(hit.get("ref_id", "")),
                        score=hit.get("score"),
                        snippet=str(hit.get("snippet", "")),
                    )
                )
            params = dict(params_in)
            params["depth"] = self._depth
            trace.retrieval.append(
                RetrievalEvent(
                    op=str(event.get("op", "")),
                    query=event.get("query"),
                    params=params,
                    hits=hits,
                )
            )

        note_oracle = None
        note_yesno = None
        hidden_note = self._options.hidden_note
        if hidden_note is not None:
            allow_insecure = os.environ.get("RLM_ALLOW_INSECURE_NOTE") == "1"
            yesno_retries = max(1, int(os.environ.get("LLM_YESNO_MAX_RETRIES", "4")))
            system_line = "Answer yes or no only."
            try:
                note_oracle = NoteOracleClient.from_llm(
                    self._sub_llm, hidden_note, max_retries=yesno_retries
                )

                def _oracle_note_yesno(question: str) -> str:
                    prompt = f"{question} Note: {hidden_note}"
                    _bump_subcall(len(prompt), len(system_line))
                    answer = note_oracle.note_yesno(question)
                    trace.subcalls.append(
                        SubcallTrace(
                            prompt_chars=len(prompt),
                            system_chars=len(system_line),
                            response_chars=len(answer or ""),
                        )
                    )
                    trace.subcall_output_chars += len(answer or "")
                    return answer

                note_yesno = _oracle_note_yesno
            except NoteOracleError as exc:
                if not allow_insecure:
                    raise RuntimeError(
                        "Hidden note isolation requires OpenAICompatibleClient or "
                        "OpenAIResponsesClient for the sub-LLM. "
                        "Set RLM_ALLOW_INSECURE_NOTE=1 to use in-process note_yesno "
                        "(results may be compromised)."
                    ) from exc

                def _insecure_note_yesno(question: str) -> str:
                    prompt = f"{question} Note: {hidden_note}"
                    last_response = ""
                    for attempt in range(yesno_retries):
                        if attempt == 0:
                            response = llm_query(prompt, system=system_line)
                        else:
                            response = llm_query(
                                f"Answer yes or no only. {prompt}",
                                system=system_line,
                            )
                        last_response = response or ""
                        normalized = _normalize_yesno(last_response)
                        if normalized is not None:
                            return normalized
                    raise RuntimeError(
                        f"note_yesno failed after {yesno_retries} retries: {last_response!r}"
                    )

                note_yesno = _insecure_note_yesno

        rlm_query_state = {"count": 0}

        def rlm_query(question: str, sub_context: Any, depth_limit: int = 1) -> str:
            rlm_query_state["count"] += 1
            trace.rlm_queries = rlm_query_state["count"]
            if depth_limit < 1:
                raise RuntimeError("rlm_query depth_limit must be >= 1")
            effective_limit = depth_limit
            if self._depth_limit is not None:
                effective_limit = min(effective_limit, self._depth_limit)
            if self._depth >= effective_limit:
                raise RuntimeError("rlm_query depth limit exceeded")
            cache_key = None
            if self._options.cache_enabled and self._cache is not None:
                payload = json.dumps(
                    {
                        "q": question,
                        "ctx": str(sub_context),
                        "depth": effective_limit,
                        "protocol": self._options.protocol,
                    },
                    sort_keys=True,
                )
                cache_key = hashlib.sha1(payload.encode("utf-8")).hexdigest()
                cached = self._cache.get(cache_key)
                if cached is not None:
                    trace.rlm_cache_hits += 1
                    return cached
                trace.rlm_cache_misses += 1
            child = RLM(
                root_llm=self._sub_llm,
                sub_llm=self._sub_llm,
                options=self._options,
                _depth=self._depth + 1,
                _depth_limit=effective_limit,
                _budget=budget,
                _trace=trace,
                _cache=self._cache,
            )
            result = child.answer(question, sub_context)
            if cache_key and self._cache is not None:
                self._cache[cache_key] = result
            return result

        repl = ReplEnvironment(
            context=context,
            llm_query=llm_query,
            max_output_chars=self._options.max_output_chars,
            note_yesno=note_yesno,
            rlm_query=rlm_query,
            trace_hook=trace_hook,
            repl_timeout_s=self._options.repl_timeout_s,
            repl_memory_mb=self._options.repl_memory_mb,
            repl_cpu_seconds=self._options.repl_cpu_seconds,
            enable_ctx=self._options.enable_ctx,
            max_snippet_chars=self._options.max_trace_snippet_chars,
            cache_enabled=self._options.cache_enabled,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        saw_repl = False
        repl_block_index = 0
        try:
            for _ in range(self._options.max_steps):
                trace.steps += 1
                model_input = sum(len(str(m.get("content", ""))) for m in messages)
                budget["model_input"] += model_input
                trace.model_input_chars = budget["model_input"]
                assistant = self._root_llm.complete(messages)
                trace.assistant_messages.append(
                    {"depth": self._depth, "content": assistant}
                )
                messages.append({"role": "assistant", "content": assistant})

                repl_blocks, final, parse_error = _parse_response(
                    assistant, self._options.protocol
                )
                if parse_error:
                    if self._options.retry_on_invalid:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Your response must be valid JSON with keys "
                                    "`repl` and `final` or `final_var`, and contain no extra text."
                                ),
                            }
                        )
                        continue
                    raise RuntimeError(f"Invalid JSON response: {parse_error}")
                if repl_blocks:
                    saw_repl = True

                    outputs = []
                    for block in repl_blocks:
                        repl_block_index += 1
                        result = repl.exec(block)
                        output = str(result.get("output") or "")
                        outputs.append(
                            {
                                "index": repl_block_index,
                                "output": output,
                                "truncated": bool(result.get("truncated")),
                                "error": result.get("error"),
                            }
                        )
                        output_len = len(output)
                        budget["repl_output"] += output_len
                        trace.repl_output_chars = budget["repl_output"]
                        trace.repl_blocks.append(
                            ReplBlockTrace(
                                index=repl_block_index,
                                code_chars=len(block),
                                output_chars=output_len,
                                truncated=bool(result.get("truncated")),
                                error=result.get("error"),
                            )
                        )
                        if max_total_repl_output is not None and budget["repl_output"] > max_total_repl_output:
                            raise RuntimeError("Total REPL output budget exceeded")

                    formatted_outputs = _format_repl_outputs(
                        outputs, self._options.protocol
                    )
                    budget["surfaced"] += len(formatted_outputs)
                    trace.surfaced_chars = budget["surfaced"]
                    if max_total_surfaced is not None and budget["surfaced"] > max_total_surfaced:
                        raise RuntimeError("Total surfaced chars budget exceeded")
                    if self._options.log_repl_outputs:
                        print(formatted_outputs)
                    messages.append({"role": "user", "content": formatted_outputs})

                if final is not None:
                    if self._options.require_repl and not saw_repl:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "You must use the REPL before answering. "
                                    "Run at least one ```repl``` block to inspect context, "
                                    "then respond with FINAL(...)."
                                ),
                            }
                        )
                        continue
                    if self._options.min_sub_calls and subcall_state["count"] < self._options.min_sub_calls:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"You must call llm_query at least {self._options.min_sub_calls} times "
                                    f"before answering. Current count: {subcall_state['count']}. "
                                    "Run a ```repl``` block that calls llm_query, then respond with FINAL(...)."
                                ),
                            }
                        )
                        continue
                    if self._options.min_rlm_queries and rlm_query_state["count"] < self._options.min_rlm_queries:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"You must call rlm_query at least {self._options.min_rlm_queries} times "
                                    f"before answering. Current count: {rlm_query_state['count']}. "
                                    "Run a ```repl``` block that calls rlm_query, then respond with FINAL(...)."
                                ),
                            }
                        )
                        continue
                    try:
                        result = _resolve_final(final, repl)
                        trace.termination_reason = "final"
                        return result
                    except RuntimeError as exc:
                        if self._options.retry_on_invalid:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "Your FINAL_VAR referenced a variable that does not exist. "
                                        "Run a ```repl``` block to define it, then respond with FINAL_VAR(...). "
                                        f"Error: {exc}"
                                    ),
                                }
                            )
                            continue
                        trace.termination_reason = f"error:{exc}"
                        raise

                if not repl_blocks:
                    if self._options.retry_on_invalid:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Your response must include either a ```repl``` block "
                                    "or a FINAL(...) / FINAL_VAR(...) line. Try again."
                                ),
                            }
                        )
                        continue
                    trace.termination_reason = "error:no_repl_or_final"
                    raise RuntimeError("No repl block or FINAL found in assistant output")

            trace.termination_reason = "error:max_steps"
            raise RuntimeError("Max steps reached without FINAL")
        except Exception as exc:
            if trace.termination_reason is None:
                trace.termination_reason = f"error:{exc}"
            raise
        finally:
            repl.close()
            if note_oracle is not None:
                note_oracle.close()


def _describe_context(context: Any) -> Tuple[str, int, List[int]]:
    if isinstance(context, (list, tuple)):
        lengths = [len(str(item)) for item in context]
        total = sum(lengths)
        return "list[str]", total, lengths
    text = str(context)
    return "string", len(text), [len(text)]


def _extract_repl_blocks(text: str) -> List[str]:
    return [match.strip() for match in RE_REPL_BLOCK.findall(text)]


def _extract_final(text: str) -> Optional[Tuple[str, str]]:
    without_blocks = RE_REPL_BLOCK.sub("", text)
    final: Optional[Tuple[str, str]] = None
    for line in without_blocks.splitlines():
        match_var = RE_FINAL_VAR_LINE.match(line)
        if match_var:
            final = ("var", match_var.group(1).strip())
            continue
        match_final = RE_FINAL_LINE.match(line)
        if match_final:
            final = ("direct", match_final.group(1).strip())
    return final


def _parse_response(
    text: str, protocol: str
) -> Tuple[List[str], Optional[Tuple[str, str]], Optional[str]]:
    if protocol == "json":
        obj, error = _parse_json_response(text)
        if obj is None:
            return [], None, error or "invalid json"
        return _extract_json_repl(obj), _extract_json_final(obj), None
    return _extract_repl_blocks(text), _extract_final(text), None


def _parse_json_response(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        return None, str(exc)
    if not isinstance(parsed, dict):
        return None, "JSON must be an object"
    return parsed, None


def _extract_json_repl(obj: Dict[str, Any]) -> List[str]:
    repl = obj.get("repl")
    if repl is None:
        return []
    if isinstance(repl, str):
        return [repl]
    if isinstance(repl, list):
        blocks = []
        for item in repl:
            if isinstance(item, dict) and "code" in item:
                blocks.append(str(item["code"]))
            elif isinstance(item, str):
                blocks.append(item)
        return blocks
    return []


def _extract_json_final(obj: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if "final_var" in obj:
        return ("var", _strip_final_wrappers(str(obj.get("final_var", "")).strip()))
    if "final" in obj:
        return ("direct", _strip_final_wrappers(str(obj.get("final", "")).strip()))
    return None


def _strip_final_wrappers(value: str) -> str:
    match_final = RE_FINAL_LINE.match(value)
    if match_final:
        return match_final.group(1).strip()
    match_var = RE_FINAL_VAR_LINE.match(value)
    if match_var:
        return match_var.group(1).strip()
    return value


def _resolve_final(final: Tuple[str, str], repl: ReplEnvironment) -> str:
    kind, value = final
    if kind == "direct":
        return value
    try:
        resolved = repl.get_var(value)
    except KeyError as exc:
        raise RuntimeError(str(exc))
    return str(resolved)


def _normalize_yesno(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    cleaned = text.strip().lower()
    if cleaned.startswith("y"):
        return "yes"
    if cleaned.startswith("n"):
        return "no"
    return None


def _format_repl_outputs(outputs: List[Dict[str, Any]], protocol: str) -> str:
    if protocol == "json":
        payload = {
            "repl_output": [
                {
                    "index": output.get("index"),
                    "output": output.get("output"),
                    "truncated": output.get("truncated"),
                    "error": output.get("error"),
                }
                for output in outputs
            ]
        }
        return json.dumps(payload)
    lines = ["REPL OUTPUT:"]
    for output in outputs:
        idx = output.get("index")
        lines.append(f"[block {idx}]")
        lines.append(str(output.get("output", "")))
    return "\n".join(lines)
