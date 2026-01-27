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
    require_repl: bool = False
    retry_on_invalid: bool = False
    log_repl_outputs: bool = False
    min_sub_calls: int = 0


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

        saw_repl = False
        for _ in range(self._options.max_steps):
            assistant = self._root_llm.complete(messages)
            messages.append({"role": "assistant", "content": assistant})

            repl_blocks = _extract_repl_blocks(assistant)
            if repl_blocks:
                saw_repl = True

                outputs = []
                for block in repl_blocks:
                    outputs.append(repl.exec(block))

                formatted_outputs = _format_repl_outputs(outputs)
                if self._options.log_repl_outputs:
                    print(formatted_outputs)
                messages.append({"role": "user", "content": formatted_outputs})

            final = _extract_final(assistant)
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
                try:
                    return _resolve_final(final, repl)
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
                raise RuntimeError("No repl block or FINAL found in assistant output")

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
