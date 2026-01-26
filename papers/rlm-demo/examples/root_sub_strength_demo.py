from rlm_demo import LLMClient, RLM


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
    query = "For claim C100, is it eligible and what is the payout?"

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

    weak_answer = run_case(
        "Weak root + weak sub", weak_root, weak_sub, query, context
    )
    strong_answer = run_case(
        "Strong root + strong sub", strong_root, strong_sub, query, context
    )

    if weak_answer != expected and strong_answer == expected:
        print("Result: weak models fail, strong models succeed.")


if __name__ == "__main__":
    main()
