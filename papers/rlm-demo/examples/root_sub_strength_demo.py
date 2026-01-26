import os

from rlm_demo import LLMClient, OpenAIResponsesClient, RLM


class ScriptedLLM(LLMClient):
    def __init__(self, replies):
        self._replies = list(replies)

    def complete(self, messages):
        if not self._replies:
            raise RuntimeError("ScriptedLLM ran out of replies")
        return self._replies.pop(0)


class StaticLLM(LLMClient):
    def __init__(self, reply):
        self._reply = reply

    def complete(self, messages):
        return self._reply


def expected_answer(note, plan, amount):
    if "out of network" in note.lower():
        eligible = False
    else:
        eligible = "emergency towing" in note.lower()
    payout = 0
    if eligible:
        rate = 0.8 if plan == "Gold" else 0.5
        payout = int(float(amount) * rate)
    return f"eligible={eligible}, payout={payout}"


def run_case(label, root_replies, sub_reply, query, context):
    rlm = RLM(root_llm=ScriptedLLM(root_replies), sub_llm=StaticLLM(sub_reply))
    answer = rlm.answer(query, context)
    print(f"{label} answer: {answer}")
    return answer


def supports_reasoning_model(model_name):
    if not model_name:
        return False
    name = model_name.strip().lower()
    return name.startswith("gpt-5") or name.startswith("o")


def run_live_case(label, root_model, sub_model, query, context):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY to run live model experiments.")
    base_url = os.environ.get("OPENAI_BASE_URL")
    text_verbosity = os.environ.get("OPENAI_TEXT_VERBOSITY", "medium")

    def normalize_effort(value):
        value = (value or "").strip()
        return value if value else None

    root_effort = normalize_effort(os.environ.get("ROOT_MODEL_EFFORT"))
    sub_effort = normalize_effort(os.environ.get("SUB_MODEL_EFFORT"))

    root_client = OpenAIResponsesClient(
        api_key=api_key,
        base_url=base_url,
        model=root_model,
        reasoning_effort=root_effort if supports_reasoning_model(root_model) else None,
        text_verbosity=text_verbosity,
    )
    sub_client = OpenAIResponsesClient(
        api_key=api_key,
        base_url=base_url,
        model=sub_model,
        reasoning_effort=sub_effort if supports_reasoning_model(sub_model) else None,
        text_verbosity=text_verbosity,
    )
    rlm = RLM(root_llm=root_client, sub_llm=sub_client)
    answer = rlm.answer(query, context)
    print(f"{label} ({root_model} / {sub_model}) answer: {answer}")
    return answer


def main():
    policy = """\
Rules:
1) Eligible if plan == Gold OR (plan == Silver and note implies emergency towing).
2) If note indicates out-of-network, ineligible.
Payout: Gold = 0.8 * amount, Silver = 0.5 * amount.
"""
    claims_csv = """\
id,plan,amount,note
C100,Silver,1000,includes emergency towing; in-network
C101,Gold,2000,in-network
C102,Silver,1500,out of network; emergency towing
"""
    context = [policy, claims_csv]
    query = (
        "Use the REPL to parse the CSV and policy. If needed, call llm_query to "
        "interpret the note. For claim C100, is it eligible and what is the payout?"
    )

    weak_root = [
        """```repl
policy, claims_csv = context
lines = [line.strip() for line in claims_csv.splitlines() if line.strip()]
header = [h.strip() for h in lines[0].split(",")]
rows = []
for line in lines[1:]:
    parts = [p.strip() for p in line.split(",")]
    rows.append(dict(zip(header, parts)))
row = [r for r in rows if r["id"] == "C100"][0]
note = row["note"]
if "out of network" in note.lower():
    eligible = False
else:
    ans = llm_query(
        f"Answer yes or no: does this note imply emergency towing? Note: {note}"
    )
    eligible = ans.strip().lower() == "yes"
payout = 0
if eligible:
    rate = 0.8 if row["plan"] == "Gold" else 0.5
    payout = int(float(row["amount"]) * rate)
answer = f"eligible={eligible}, payout={payout}"
print(answer)
```""",
        "FINAL_VAR(answer)",
    ]

    strong_root = [
        """```repl
policy, claims_csv = context
lines = [line.strip() for line in claims_csv.splitlines() if line.strip()]
header = [h.strip() for h in lines[0].split(",")]
rows = []
for line in lines[1:]:
    parts = [p.strip() for p in line.split(",")]
    rows.append(dict(zip(header, parts)))
row = [r for r in rows if r["id"] == "C100"][0]
note = row["note"]
if "out of network" in note.lower():
    eligible = False
else:
    ans = llm_query(
        f"Answer yes or no: does this note imply emergency towing? Note: {note}"
    )
    ans_norm = ans.strip().lower()
    eligible = ans_norm.startswith("y") or "yes" in ans_norm
payout = 0
if eligible:
    rate = 0.8 if row["plan"] == "Gold" else 0.5
    payout = int(float(row["amount"]) * rate)
answer = f"eligible={eligible}, payout={payout}"
print(answer)
```""",
        "FINAL_VAR(answer)",
    ]

    weak_sub = "Yes, it includes emergency towing."
    strong_sub = "yes"

    expected = expected_answer(
        note="includes emergency towing; in-network",
        plan="Silver",
        amount="1000",
    )
    print("Expected:", expected)

    use_scripted = os.environ.get("USE_SCRIPTED_DEMO") == "1"
    if use_scripted:
        weak_answer = run_case(
            "Weak root + weak sub", weak_root, weak_sub, query, context
        )
        strong_answer = run_case(
            "Strong root + strong sub", strong_root, strong_sub, query, context
        )
    else:
        weak_root_model = os.environ.get("WEAK_ROOT_MODEL", "gpt-4.1-nano")
        weak_sub_model = os.environ.get("WEAK_SUB_MODEL", "gpt-4.1-nano")
        strong_root_model = os.environ.get("STRONG_ROOT_MODEL", "gpt-5.2")
        strong_sub_model = os.environ.get("STRONG_SUB_MODEL", "gpt-5.2")

        weak_answer = run_live_case(
            "Weak root + weak sub", weak_root_model, weak_sub_model, query, context
        )
        strong_answer = run_live_case(
            "Strong root + strong sub",
            strong_root_model,
            strong_sub_model,
            query,
            context,
        )

    if weak_answer != expected and strong_answer == expected:
        print("Result: weak models fail, strong models succeed.")


if __name__ == "__main__":
    main()
