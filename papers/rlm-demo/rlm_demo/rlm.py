import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .llm import LLMClient
from .prompts import render_system_prompt
from .repl import ReplEnvironment

RE_REPL_BLOCK = re.compile(r"```repl\s*(.*?)```", re.DOTALL | re.IGNORECASE)
RE_FINAL_VAR = re.compile(r"FINAL_VAR\((.*?)\)", re.DOTALL)
RE_FINAL = re.compile(r"FINAL\((.*?)\)", re.DOTALL)


@dataclass
class RLMOptions:
    max_steps: int = 8
    max_output_chars: int = 4000
    max_sub_calls: int = 32
    include_cost_hint: bool = True


class RLM:
    def __init__(
        self,
        root_llm: LLMClient,
        sub_llm: Optional[LLMClient] = None,
        options: Optional[RLMOptions] = None,
    ) -> None:
        self._root_llm = root_llm
        self._sub_llm = sub_llm or root_llm
        self._options = options or RLMOptions()

    def answer(self, query: str, context: Any) -> str:
        context_type, total_length, chunk_lengths = _describe_context(context)
        system_prompt = render_system_prompt(
            context_type=context_type,
            context_total_length=total_length,
            context_lengths=chunk_lengths,
            include_cost_hint=self._options.include_cost_hint,
        )

        subcall_state = {"count": 0}

        def llm_query(prompt: Any, system: Optional[str] = None) -> str:
            subcall_state["count"] += 1
            if subcall_state["count"] > self._options.max_sub_calls:
                raise RuntimeError("llm_query call limit exceeded")
            messages: List[Dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": str(system)})
            messages.append({"role": "user", "content": str(prompt)})
            return self._sub_llm.complete(messages)

        repl = ReplEnvironment(
            context=context,
            llm_query=llm_query,
            max_output_chars=self._options.max_output_chars,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        for _ in range(self._options.max_steps):
            assistant = self._root_llm.complete(messages)
            messages.append({"role": "assistant", "content": assistant})

            final = _extract_final(assistant)
            if final is not None:
                return _resolve_final(final, repl)

            repl_blocks = _extract_repl_blocks(assistant)
            if not repl_blocks:
                raise RuntimeError("No repl block or FINAL found in assistant output")

            outputs = []
            for block in repl_blocks:
                outputs.append(repl.exec(block))

            messages.append({"role": "user", "content": _format_repl_outputs(outputs)})

        raise RuntimeError("Max steps reached without FINAL")


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
    match_var = RE_FINAL_VAR.search(without_blocks)
    if match_var:
        return ("var", match_var.group(1).strip())
    match_final = RE_FINAL.search(without_blocks)
    if match_final:
        return ("direct", match_final.group(1).strip())
    return None


def _resolve_final(final: Tuple[str, str], repl: ReplEnvironment) -> str:
    kind, value = final
    if kind == "direct":
        return value
    try:
        resolved = repl.get_var(value)
    except KeyError as exc:
        raise RuntimeError(str(exc))
    return str(resolved)


def _format_repl_outputs(outputs: List[str]) -> str:
    lines = ["REPL OUTPUT:"]
    for idx, output in enumerate(outputs, start=1):
        lines.append(f"[block {idx}]")
        lines.append(output)
    return "\n".join(lines)