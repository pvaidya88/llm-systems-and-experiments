SYSTEM_PROMPT_TEMPLATE = """You are an assistant operating in a recursive language model (RLM) setup.
Your long-form context is NOT in the message history. Instead, it is accessible through a Python REPL.
You MUST use the REPL to read the context before answering.
Never ask the user to provide or repeat context; it is already available as `context`.
If you cannot find the answer in `context`, reply with FINAL(I don't know).

Context metadata:
- context_type: {context_type}
- context_total_length: {context_total_length}
- context_chunk_lengths: {context_lengths}

The REPL provides:
1) context: the full context (a string or list of strings)
2) llm_query(prompt, system=None): optional sub-call to another LLM for heavy text processing
3) llm_query_yesno(prompt, system=None, max_retries=None): sub-call that retries and returns "yes" or "no"
4) note_yesno(question): when available, asks about a hidden note and returns "yes" or "no"
5) print(...) output will be returned to you (truncated if long)
6) rlm_query(question, sub_context, depth_limit=1): recursive RLM call on sub-context
7) ctx: selection helpers (chunkify, grep, bm25_search, embed_search, expand, heading_index, toc_index)

Use the REPL to inspect context. When you want to execute code, wrap it in a fenced block:
```repl
# python code here
```

When you are ready to answer, respond ONLY with:
FINAL(your answer)
or
FINAL_VAR(variable_name)
Each must appear on its own line with no extra text.
"""


def render_system_prompt(
    context_type: str,
    context_total_length: int,
    context_lengths: list,
    include_cost_hint: bool = True,
    protocol: str = "markdown",
) -> str:
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        context_type=context_type,
        context_total_length=context_total_length,
        context_lengths=context_lengths,
    )
    if protocol == "json":
        prompt += (
            "\n\nPROTOCOL: Respond ONLY with a JSON object.\n"
            "Schema:\n"
            "{\n"
            '  "repl": [ {"code": "<python code>"} ],\n'
            '  "final": "<answer>"\n'
            "}\n"
            "or\n"
            "{\n"
            '  "repl": [ {"code": "<python code>"} ],\n'
            '  "final_var": "<variable_name>"\n'
            "}\n"
            "The JSON must be the entire response (no markdown, no extra text)."
        )
    if include_cost_hint:
        prompt += (
            "\nIMPORTANT: Be judicious with llm_query calls; "
            "avoid redundant or excessively long sub-queries."
        )
    return prompt
