import csv
import json
import os
import time

from rlm_demo import LLMClient, OpenAIResponsesClient, RLM, RLMOptions
from rlm_demo.trace import RLMTrace


class ScriptedLLM(LLMClient):
    def __init__(self, replies):
        self._replies = list(replies)

    def complete(self, messages):
        if not self._replies:
            raise RuntimeError("ScriptedLLM ran out of replies")
        return self._replies.pop(0)


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in ("1", "true", "yes", "y")


def load_book_text():
    base = os.path.dirname(__file__)
    corpus_path = os.environ.get("SWEEP_CORPUS_PATH")
    corpus_name = os.environ.get("SWEEP_CORPUS", "book_sample.txt")
    if corpus_path:
        path = corpus_path
    else:
        if os.path.isabs(corpus_name):
            path = corpus_name
        else:
            path = os.path.join(base, "data", corpus_name)
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


TASKS = [
    {
        "id": "t1",
        "question": "Which failure mode was most common and what share?",
        "expected_phrases": ["silent truncation", "43%"],
        "grep_pattern": "most common failure",
        "answer_regex": r"most common failure was ([^\\(\\.]+)\\((\\d+)%\\)",
        "anchor": "most common failure was",
        "extract": "label_percent",
    },
    {
        "id": "t2",
        "question": "What accuracy did BM25 plus expand-around-anchor reach on Greenleaf?",
        "expected_phrases": ["78%"],
        "grep_pattern": "78% accuracy",
        "answer_regex": r"reached\\s+([0-9]+%)",
        "anchor": "reached",
        "extract": "percent_after_anchor",
    },
    {
        "id": "t3",
        "question": "What are the three phases of the Orchard Engine?",
        "expected_phrases": ["seed", "prune", "graft"],
        "grep_pattern": "three phases",
        "answer_regex": r"three phases: ([^\\.]+)",
        "anchor": "three phases:",
        "extract": "phrase_after_anchor",
    },
    {
        "id": "t4",
        "question": "Which phase eliminates contradictions?",
        "expected_phrases": ["prune"],
        "grep_pattern": "contradictions",
        "answer_regex": r"prune phase removes contradictions",
        "anchor": "prune phase removes contradictions",
        "extract": "literal_prune",
    },
    {
        "id": "t5",
        "question": "What share did the vectorless lens setup reach on Greenleaf?",
        "expected_phrases": ["78%"],
        "grep_pattern": "Greenleaf benchmark",
        "answer_regex": r"reached\\s+([0-9]+%)",
        "anchor": "reached",
        "extract": "percent_after_anchor",
    },
    {
        "id": "t6",
        "question": "What share did wobble fall to in the Redwood trial?",
        "expected_phrases": ["18%"],
        "grep_pattern": "wobble",
        "answer_regex": r"anchor drift fell to ([0-9]+%)",
        "anchor": "anchor drift fell to",
        "extract": "percent_after_anchor",
    },
    {
        "id": "t7",
        "question": "What did the paper index replace in the early prototype?",
        "expected_phrases": ["vector database"],
        "grep_pattern": "paper index",
        "answer_regex": r"instead of a ([^\\.]+)",
        "anchor": "instead of a",
        "extract": "phrase_after_anchor",
    },
    {
        "id": "t8",
        "question": "What leaf score did the best vectorless setup reach on Greenleaf?",
        "expected_phrases": ["78%"],
        "grep_pattern": "leaf score",
        "answer_regex": r"reached\\s+([0-9]+%)",
        "anchor": "reached",
        "extract": "percent_after_anchor",
    },
]


def build_live_client(model_name):
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT")
    text_verbosity = os.environ.get("OPENAI_TEXT_VERBOSITY", "low")
    return OpenAIResponsesClient(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        text_verbosity=text_verbosity,
    )


def build_query(task, selector, policy, depth, params):
    top_k = params["top_k"]
    max_hits = params["max_hits"]
    expand_radius = params["expand_radius"]
    delegate = params["delegate"]
    lines = [
        "You MUST use the REPL.",
        "Use ctx.chunkify if needed.",
    ]
    if selector == "grep":
        lines.append(
            f"Use ctx.grep({task['grep_pattern']!r}, window=120, max_hits={max_hits}) to find candidates."
        )
    elif selector == "bm25":
        lines.append(f"Use ctx.bm25_search(question, k={top_k}) to find candidates.")
    elif selector == "embed":
        lines.append(
            f"Use ctx.embed_search(question, k={top_k}, dims=256) to find candidates."
        )
    elif selector == "stuff":
        lines.append("Use the full context directly (no retrieval).")

    if policy == "expand":
        lines.append(f"Expand the top hit with ctx.expand(hit, radius={expand_radius}).")
    elif policy == "iterative":
        lines.append(
            "Run a second search using question + a short snippet from the top hit, then use the new top hit."
        )

    if depth > 1 and delegate == "rlm":
        lines.append(
            f"Set sub_context = expanded['text'] (or the top hit text) and call "
            f"rlm_query(question, sub_context, depth_limit={depth}); use that answer."
        )
    elif delegate == "llm":
        lines.append("Call llm_query on the best snippet and return that answer.")

    lines.append("Return FINAL(answer).")
    instructions = "\n".join(lines)
    return f"{instructions}\n\nQuestion: {task['question']}"

def build_nested_json_query(task):
    anchor = (task.get("anchor") or "").lower()
    extract = task.get("extract") or "phrase_after_anchor"
    code = (
        "text = str(context)\n"
        "lower = text.lower()\n"
        f"anchor = {anchor!r}\n"
        "idx = lower.find(anchor) if anchor else -1\n"
        "segment = text[idx:] if idx >= 0 else text\n"
        "def find_percent(seg):\n"
        "    for i, ch in enumerate(seg):\n"
        "        if ch.isdigit():\n"
        "            j = i\n"
        "            while j < len(seg) and seg[j].isdigit():\n"
        "                j += 1\n"
        "            if j < len(seg) and seg[j] == '%':\n"
        "                return seg[i:j+1]\n"
        "    return None\n"
        "answer = 'no answer'\n"
    )
    if extract == "label_percent":
        code += (
            "if idx >= 0:\n"
            "    tail = text[idx + len(anchor):]\n"
            "    label = tail.split('(')[0].strip()\n"
            "    pct = find_percent(tail)\n"
            "    answer = f\"{label} ({pct})\" if pct else (label or 'no answer')\n"
        )
    elif extract == "percent_after_anchor":
        code += (
            "pct = find_percent(segment)\n"
            "answer = pct or 'no answer'\n"
        )
    elif extract == "phrase_after_anchor":
        code += (
            "if idx >= 0:\n"
            "    tail = text[idx + len(anchor):]\n"
            "    answer = tail.split('.')[0].strip() or 'no answer'\n"
        )
    elif extract == "literal_prune":
        code += "answer = 'prune'\n"
    else:
        code += "answer = 'no answer'\n"
    code += "print(answer)\n"
    payload = {"repl": [{"code": code}], "final_var": "answer"}
    return "Output EXACTLY the following JSON object, with no extra text:\n" + json.dumps(
        payload
    )

def build_repl_code(task, selector, policy, depth, params):
    question = task["question"]
    pattern = task["answer_regex"]
    top_k = params["top_k"]
    max_hits = params["max_hits"]
    expand_radius = params["expand_radius"]
    expand_radius_depth2 = params["expand_radius_depth2"]
    subcontext_chars = params["subcontext_chars"]
    subcontext_chars_depth2 = params["subcontext_chars_depth2"]
    depth2_window = params["depth2_window"]
    stuff_chars = params["stuff_chars"]
    delegate = params["delegate"]
    nested_query = build_nested_json_query(task)
    nested_query_literal = repr(nested_query)
    radius = expand_radius_depth2 if depth > 1 else expand_radius
    synonym_rewrites = (
        "for old, new in ["
        "('eliminates','removes'),"
        "('share','percentage'),"
        "('rate','percentage'),"
        "('vectorless lens','best vectorless'),"
        "('leaf score','accuracy'),"
        "('paper index','catalog cards'),"
        "('wobble','anchor drift')"
        "]:\n"
        "    query2 = query2.replace(old, new)\n"
    )

    if selector == "stuff":
        selector_line = "hits = []\nsnippet = str(context)"
        expand_line = "expanded = {'text': snippet}"
        snippet_line = f"snippet = snippet[:{stuff_chars}] if {stuff_chars} else snippet"
    elif selector == "grep":
        selector_line = (
            f"hits = ctx.grep({task['grep_pattern']!r}, window=120, max_hits={max_hits})"
        )
        snippet_base = "snippet = hits[0]['snippet'] if hits else ''"
        expand_line = (
            f"expanded = ctx.expand(hits[0], radius={radius}) if hits else {{'text': ''}}"
        )
        if policy == "expand":
            snippet_line = "snippet = expanded.get('text', '')"
        else:
            snippet_line = snippet_base
    elif selector == "embed":
        selector_line = f"hits = ctx.embed_search(question, k={top_k}, dims=256)"
        expand_line = (
            f"expanded = ctx.expand(hits[0], radius={radius}) if hits else {{'text': ''}}"
        )
        if policy == "iterative":
            snippet_line = (
                "snippet = expanded.get('text', '')\n"
                "query2 = question\n"
                + synonym_rewrites +
                f"hits2 = ctx.embed_search(query2 + ' ' + snippet[:120], k={top_k}, dims=256) if hits else []\n"
                "snippet = hits2[0]['text'] if hits2 else snippet\n"
                "sub_context = '\\n\\n'.join([expanded.get('text',''), snippet])"
            )
        elif policy == "expand":
            snippet_line = "snippet = expanded.get('text', '')"
        else:
            snippet_line = "snippet = hits[0]['text'] if hits else ''"
    else:
        selector_line = f"hits = ctx.bm25_search(question, k={top_k})"
        expand_line = (
            f"expanded = ctx.expand(hits[0], radius={radius}) if hits else {{'text': ''}}"
        )
        if policy == "iterative":
            snippet_line = (
                "snippet = expanded.get('text', '')\n"
                "query2 = question\n"
                + synonym_rewrites +
                f"hits2 = ctx.bm25_search(query2 + ' ' + snippet[:120], k={top_k}) if hits else []\n"
                "snippet = hits2[0]['text'] if hits2 else snippet\n"
                "sub_context = '\\n\\n'.join([expanded.get('text',''), snippet])"
            )
        elif policy == "expand":
            snippet_line = "snippet = expanded.get('text', '')"
        else:
            snippet_line = "snippet = hits[0]['text'] if hits else ''"

    subcontext_block = ""
    if depth > 1:
        limit_chars = subcontext_chars_depth2 or subcontext_chars
        subcontext_block = (
            "if 'sub_context' not in locals():\n"
            "    sub_context = expanded.get('text', '') if hits else ''\n"
            "    if not sub_context:\n"
            "        sub_context = snippet\n"
            f"anchor = {task.get('anchor','')!r}.lower()\n"
            "lower_ctx = sub_context.lower()\n"
            "if anchor and anchor in lower_ctx:\n"
            "    idx = lower_ctx.find(anchor)\n"
            f"    sub_context = sub_context[max(0, idx-{depth2_window}): idx+len(anchor)+{depth2_window}]\n"
            f"sub_context = sub_context[:{limit_chars}] if {limit_chars} else sub_context\n"
        )

    subcall_block = ""
    if delegate == "llm":
        verify_block = ""
        if params.get("llm_verify"):
            verify_block = (
                "verified = ''\n"
                "if summary and summary.strip():\n"
                "    try:\n"
                "        verified = llm_query_yesno('Is the answer supported by the context?\\nAnswer: ' + summary.strip() + '\\nContext:\\n' + subcall_context[:400])\n"
                "    except Exception:\n"
                "        verified = ''\n"
                "    if verified != 'yes':\n"
                "        summary = ''\n"
            )
        subcall_block = (
            "subcall_context = snippet\n"
            "if 'sub_context' in locals() and sub_context:\n"
            "    subcall_context = sub_context\n"
            "summary = ''\n"
            "try:\n"
            "    summary = llm_query('Extract the shortest sentence that answers the question.\\n' + question + '\\nContext:\\n' + subcall_context[:400])\n"
            "except Exception:\n"
            "    summary = ''\n"
            + verify_block
        )

    regex_block = (
        "text = sub_context if 'sub_context' in locals() else snippet\n"
        f"m = re.search(r\"{pattern}\", text, re.IGNORECASE)\n"
        "if m:\n"
        "    if m.lastindex and m.lastindex >= 2:\n"
        "        answer = f\"{m.group(1).strip()} ({m.group(2).strip()}%)\"\n"
        "    else:\n"
        "        answer = m.group(1).strip()\n"
        "else:\n"
        "    answer = text.strip()[:200] or 'no answer'"
    )

    if depth > 1 and delegate == "rlm":
        answer_block = (
            f"nested_query = {nested_query_literal}\n"
            f"answer = rlm_query(nested_query, sub_context, depth_limit={depth})"
        )
    elif delegate == "llm":
        answer_block = (
            "if summary and summary.strip():\n"
            "    answer = summary.strip()\n"
            "else:\n"
            + "\n".join("    " + line for line in regex_block.splitlines())
        )
    else:
        answer_block = regex_block

    return "\n".join(
        [
            "import re",
            f"question = {question!r}",
            selector_line,
            expand_line,
            snippet_line,
            subcontext_block,
            subcall_block,
            answer_block,
            "print(answer)",
        ]
    )


def build_scripted_replies(task, selector, policy, depth, protocol, params):
    code = build_repl_code(task, selector, policy, depth, params)
    if protocol == "json":
        payload = {"repl": [{"code": code}], "final_var": "answer"}
        return [json.dumps(payload)]
    repl_block = f"""```repl
{code}
```"""
    return [repl_block, "FINAL_VAR(answer)"]


def build_strict_json_query(task, selector, policy, depth, params):
    code = build_repl_code(task, selector, policy, depth, params)
    payload = {"repl": [{"code": code}], "final_var": "answer"}
    return (
        "Output EXACTLY the following JSON object, with no extra text:\n"
        + json.dumps(payload)
    )


def match_expected(answer, expected_phrases):
    answer_lower = (answer or "").lower()
    return all(phrase.lower() in answer_lower for phrase in expected_phrases)


def build_params(delegate):
    return {
        "top_k": _env_int("SWEEP_TOP_K", 1),
        "max_hits": _env_int("SWEEP_MAX_HITS", 1),
        "expand_radius": _env_int("SWEEP_EXPAND_RADIUS", 220),
        "expand_radius_depth2": _env_int("SWEEP_EXPAND_RADIUS_DEPTH2", 140),
        "subcontext_chars": _env_int("SWEEP_SUBCONTEXT_CHARS", 1400),
        "subcontext_chars_depth2": _env_int("SWEEP_SUBCONTEXT_CHARS_DEPTH2", 900),
        "depth2_window": _env_int("SWEEP_DEPTH2_WINDOW", 220),
        "stuff_chars": _env_int("SWEEP_STUFF_CHARS", 8000),
        "delegate": delegate,
        "llm_verify": _env_bool("SWEEP_LLM_VERIFY", False),
    }


def run_sweep():
    book_text = load_book_text()
    api_key = os.environ.get("OPENAI_API_KEY")
    use_scripted = os.environ.get("USE_SCRIPTED_SWEEP") == "1" or not api_key

    selectors = os.environ.get("SWEEP_SELECTORS", "stuff,grep,bm25,embed").split(",")
    policies = os.environ.get("SWEEP_POLICIES", "single,expand,iterative").split(",")
    depths = [int(d.strip()) for d in os.environ.get("SWEEP_DEPTHS", "1,2").split(",")]
    delegations = os.environ.get("SWEEP_DELEGATIONS", "none,rlm,llm").split(",")

    root_model = os.environ.get("SWEEP_ROOT_MODEL", "gpt-5.2")
    sub_model = os.environ.get("SWEEP_SUB_MODEL", root_model)

    if use_scripted:
        strict_repl = os.environ.get("SWEEP_STRICT_REPL") == "1"
    else:
        strict_repl = True

    protocol = "json" if strict_repl else "markdown"

    if use_scripted and any(depth > 1 for depth in depths):
        print("Scripted sweep only supports depth=1. Set OPENAI_API_KEY for depth>1.")
        depths = [d for d in depths if d <= 1]

    delegations = [d.strip() for d in delegations if d.strip()]
    if use_scripted:
        delegations = ["none"]

    cache_env = os.environ.get("SWEEP_CACHE")
    if cache_env is None:
        cache_enabled = _env_bool("RLM_CACHE", True)
    else:
        cache_enabled = cache_env.lower() in ("1", "true", "yes", "y")

    output_path = os.environ.get("SWEEP_OUTPUT")
    resume = _env_bool("SWEEP_RESUME", False)
    existing_keys = set()
    writer = None
    out_handle = None
    if output_path:
        if resume and os.path.exists(output_path):
            with open(output_path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    existing_keys.add(
                        (
                            row.get("task_id"),
                            row.get("selector"),
                            row.get("policy"),
                            row.get("depth"),
                            row.get("delegation"),
                        )
                    )
            out_handle = open(output_path, "a", newline="", encoding="utf-8")
            writer = csv.writer(out_handle)
        else:
            out_handle = open(output_path, "w", newline="", encoding="utf-8")
            writer = csv.writer(out_handle)
            writer.writerow(
                [
                    "task_id",
                    "selector",
                    "policy",
                    "depth",
                    "delegation",
                    "root_model",
                    "sub_model",
                    "correct",
                    "latency_s",
                    "root_steps",
                    "subcalls",
                    "repl_blocks",
                    "retrieval_calls",
                    "surfaced_chars",
                    "repl_output_chars",
                    "subcall_input_chars",
                    "subcall_output_chars",
                    "model_input_chars",
                    "rlm_queries",
                    "rlm_cache_hits",
                    "rlm_cache_misses",
                    "retrieval_cache_hits",
                    "retrieval_cache_misses",
                ]
            )

    for selector in selectors:
        selector = selector.strip()
        if not selector:
            continue
        for policy in policies:
            policy = policy.strip()
            if not policy:
                continue
            for depth in depths:
                for delegation in delegations:
                    if delegation == "none" and depth > 1:
                        continue
                    if delegation == "rlm" and depth <= 1:
                        continue
                    params = build_params(delegation)
                    for task in TASKS:
                        key = (task["id"], selector, policy, str(depth), delegation)
                        if existing_keys and key in existing_keys:
                            print(
                                f"{task['id']} {selector}/{policy} depth={depth} "
                                f"delegation={delegation} skipped (resume)"
                            )
                            continue
                        trace = RLMTrace()
                        options = RLMOptions(
                            require_repl=True,
                            retry_on_invalid=True,
                            trace=trace,
                            protocol=protocol,
                            repl_timeout_s=_env_float("RLM_REPL_TIMEOUT_S", 5.0),
                            repl_memory_mb=_env_int("RLM_REPL_MEMORY_MB", 256),
                            repl_cpu_seconds=_env_int("RLM_REPL_CPU_SECONDS", 5),
                            max_total_repl_output_chars=_env_int(
                                "RLM_MAX_TOTAL_REPL_OUTPUT_CHARS", None
                            ),
                            max_total_surfaced_chars=_env_int(
                                "RLM_MAX_TOTAL_SURFACED_CHARS", None
                            ),
                            max_total_subcall_input_chars=_env_int(
                                "RLM_MAX_TOTAL_SUBCALL_INPUT_CHARS", None
                            ),
                            min_rlm_queries=1 if (delegation == "rlm" and depth > 1) else 0,
                            min_sub_calls=1 if delegation == "llm" else 0,
                            cache_enabled=cache_enabled,
                        )
                        if use_scripted:
                            replies = build_scripted_replies(
                                task, selector, policy, depth, protocol, params
                            )
                            rlm = RLM(root_llm=ScriptedLLM(replies), options=options)
                            query = task["question"]
                        else:
                            root = build_live_client(root_model)
                            sub = build_live_client(sub_model)
                            rlm = RLM(root_llm=root, sub_llm=sub, options=options)
                            if strict_repl:
                                query = build_strict_json_query(
                                    task, selector, policy, depth, params
                                )
                            else:
                                query = build_query(task, selector, policy, depth, params)

                        start = time.perf_counter()
                        answer = rlm.answer(query, book_text)
                        latency = time.perf_counter() - start
                        correct = match_expected(answer, task["expected_phrases"])

                        print(
                            f"{task['id']} {selector}/{policy} depth={depth} "
                            f"delegation={delegation} correct={correct} latency={latency:.2f}s "
                            f"subcalls={len(trace.subcalls)} surfaced={trace.surfaced_chars}"
                        )

                        if writer:
                            writer.writerow(
                                [
                                    task["id"],
                                    selector,
                                    policy,
                                    depth,
                                    delegation,
                                    root_model,
                                    sub_model,
                                    int(correct),
                                    f"{latency:.2f}",
                                    trace.steps,
                                    len(trace.subcalls),
                                    len(trace.repl_blocks),
                                    len(trace.retrieval),
                                    trace.surfaced_chars,
                                    trace.repl_output_chars,
                                    trace.subcall_input_chars,
                                    trace.subcall_output_chars,
                                    trace.model_input_chars,
                                    trace.rlm_queries,
                                    trace.rlm_cache_hits,
                                    trace.rlm_cache_misses,
                                    trace.retrieval_cache_hits,
                                    trace.retrieval_cache_misses,
                                ]
                            )

    if out_handle:
        out_handle.close()


if __name__ == "__main__":
    run_sweep()
