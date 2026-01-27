import csv
import os
import random
import re
from datetime import date

from rlm_demo import LLMClient, OpenAIResponsesClient, RLM, RLMOptions

OUTPUT_PATTERN = re.compile(
    r"^eligible\s*=\s*(true|false)\s*,\s*payout\s*=\s*(\d+)\s*$",
    re.IGNORECASE,
)


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


def supports_reasoning_model(model_name):
    if not model_name:
        return False
    name = model_name.strip().lower()
    return name.startswith("gpt-5") or name.startswith("o")


def normalize_answer(text):
    if text is None:
        return None
    cleaned = text.strip()
    match = OUTPUT_PATTERN.fullmatch(cleaned)
    if not match:
        return None
    return f"eligible={match.group(1).lower()}, payout={int(match.group(2))}"


def build_query(base_query, attempt, last_error, incorrect=False):
    if attempt == 0:
        return base_query
    return (
        base_query
        + "\n\nYour previous answer was invalid. "
        + (f"Error: {last_error}. " if last_error else "")
        + ("Your previous answer was incorrect. Recompute from the policy and CSV. " if incorrect else "")
        + "Return EXACTLY: eligible=<true|false>, payout=<int>. No extra text."
    )


def _has_affirmed(note_lower, pattern):
    for match in re.finditer(pattern, note_lower):
        prefix = note_lower[: match.start()]
        tokens = re.findall(r"\b\w+\b", prefix)
        last_tokens = tokens[-3:]
        if any(
            token in ("no", "not", "never", "without", "denies", "denied", "declines", "declined")
            for token in last_tokens
        ):
            continue
        return True
    return False


def _has_affirmed_any(note_lower, patterns):
    return any(_has_affirmed(note_lower, pattern) for pattern in patterns)


def parse_note_flags(note):
    note_lower = note.lower()

    emergency_patterns = [
        r"\bemergency towing\b",
        r"\btow(?:ed|ing)?\b",
        r"\btow truck\b",
        r"\bflatbed\b",
        r"\bwrecker\b",
        r"\bwinch(?:ed|ing)?\b",
        r"\brecovery vehicle\b",
        r"\bvehicle recovery\b",
        r"\bpulled from ditch\b",
    ]
    immobile_patterns = [
        r"\bimmobile\b",
        r"\bwould not start\b",
        r"\bunable to move\b",
        r"\bdisabled vehicle\b",
        r"\bstalled on highway\b",
        r"\bcannot move\b",
    ]
    roadside_only_patterns = [
        r"\broadside assistance only\b",
        r"\broadside only\b",
        r"\bjump start only\b",
        r"\btire change only\b",
        r"\blockout only\b",
    ]
    out_of_network_patterns = [
        r"\bout of network\b",
        r"\bout-of-network\b",
        r"\boon\b",
        r"\bnon[- ]network\b",
    ]

    emergency = _has_affirmed_any(note_lower, emergency_patterns)
    immobile = _has_affirmed_any(note_lower, immobile_patterns)
    roadside_only = _has_affirmed_any(note_lower, roadside_only_patterns)
    out_of_network = _has_affirmed_any(note_lower, out_of_network_patterns)

    preauth = bool(re.search(r"\bpa-\d{3,}\b", note_lower))
    preauth = preauth or "preauth" in note_lower or "pre-auth" in note_lower
    preauth = preauth or "pre authorization" in note_lower or "pre-authorization" in note_lower

    return emergency, preauth, out_of_network, immobile, roadside_only


def compute_expected(row):
    plan = row["plan"]
    amount = int(row["amount"])
    service_date = date.fromisoformat(row["service_date"])
    filed_date = date.fromisoformat(row["filed_date"])
    note = row["note"]

    emergency, preauth, out_of_network, immobile, roadside_only = parse_note_flags(note)
    timely = (filed_date - service_date).days <= 30

    if out_of_network or roadside_only or not timely:
        eligible = False
    elif plan == "Gold":
        eligible = True
    elif plan == "Silver":
        eligible = emergency and preauth and immobile
    elif plan == "Bronze":
        eligible = emergency and immobile and amount <= 1200
    else:
        eligible = False

    payout = 0
    if eligible:
        deductibles = {"Gold": 100, "Silver": 200, "Bronze": 300}
        caps = {"Gold": 3000, "Silver": 1500, "Bronze": 800}
        payout = min(max(amount - deductibles.get(plan, 0), 0), caps.get(plan, amount))

    return f"eligible={str(eligible).lower()}, payout={payout}"


def parse_csv_rows(csv_text):
    rows = []
    reader = csv.DictReader([line for line in csv_text.splitlines() if line.strip()])
    for row in reader:
        rows.append({key.strip(): value.strip() for key, value in row.items()})
    return rows


def classify_outcome(answer, expected):
    if answer.startswith("<error:"):
        return "error"
    if answer.startswith("<invalid:"):
        return "invalid"
    if answer == expected:
        return "correct"
    return "incorrect"


def make_claims_csv(note_text):
    return f"""\
id,plan,amount,service_date,filed_date,note
C100,Silver,1800,2024-08-01,2024-08-20,{note_text}
C101,Gold,500,2024-06-10,2024-07-25,in-network; no tow needed
C102,Bronze,1000,2024-08-03,2024-08-05,Vehicle immobile; tow truck dispatched; in-network
C103,Silver,900,2024-08-02,2024-08-10,Roadside assistance only; no tow; OON
"""


def format_model_pair_label(root_model, sub_model):
    return f"root={root_model} / sub={sub_model}"


def run_case(label, root_replies, sub_reply, query, context):
    log_repl = os.environ.get("LOG_REPL_OUTPUTS") == "1"
    rlm = RLM(
        root_llm=ScriptedLLM(root_replies),
        sub_llm=StaticLLM(sub_reply),
        options=RLMOptions(
            require_repl=True,
            retry_on_invalid=True,
            log_repl_outputs=log_repl,
            max_steps=10,
        ),
    )
    answer = rlm.answer(query, context)
    normalized = normalize_answer(answer) or answer
    print(f"{label} answer: {normalized}")
    return normalized


def run_live_case(label, root_model, sub_model, query, context, expected, max_attempts):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY to run live model experiments.")
    base_url = os.environ.get("OPENAI_BASE_URL")
    text_verbosity = os.environ.get("OPENAI_TEXT_VERBOSITY", "low")

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
    log_repl = os.environ.get("LOG_REPL_OUTPUTS") == "1"
    rlm = RLM(
        root_llm=root_client,
        sub_llm=sub_client,
        options=RLMOptions(
            require_repl=True,
            retry_on_invalid=True,
            log_repl_outputs=log_repl,
            max_steps=10,
        ),
    )

    last_error = None
    last_answer = None
    last_incorrect = False
    for attempt in range(max_attempts):
        attempt_query = build_query(query, attempt, last_error, incorrect=last_incorrect)
        try:
            answer = rlm.answer(attempt_query, context)
        except Exception as exc:
            print(f"{label} error: {exc}")
            return f"<error: {exc}>"
        last_answer = answer
        normalized = normalize_answer(answer)
        if normalized:
            if normalized == expected:
                print(f"{label} answer: {normalized}")
                return normalized
            last_error = f"incorrect answer: {normalized}"
            last_incorrect = True
            continue
        last_error = f"invalid format: {answer}"
        last_incorrect = False

    print(f"{label} invalid output: {last_answer}")
    return f"<invalid: {last_answer}>"


def main():
    policy = """\
Rules:
1) Ineligible if note indicates out-of-network (e.g., "out of network", "OON", "non-network").
   Negated mentions like "not out-of-network" do NOT count as out-of-network.
2) Ineligible if note indicates "roadside assistance only" (or similar), unless explicitly negated.
   Example: "NOT roadside assistance only" does NOT trigger this rule.
3) Claim must be filed within 30 days of service_date.
4) Eligibility by plan:
   - Gold: eligible if rules 1-3 pass.
   - Silver: eligible only if note implies emergency towing, a pre-authorization code is present,
     AND the vehicle was immobile.
   - Bronze: eligible only if note implies emergency towing, the vehicle was immobile,
     AND amount <= 1200.
5) Payout = min(max(amount - deductible, 0), cap).
   - Deductible: Gold 100, Silver 200, Bronze 300.
   - Cap: Gold 3000, Silver 1500, Bronze 800.
"""

    note_variants = [
        "NOT roadside assistance only; vehicle would not start and was immobile; "
        "flatbed tow arranged after engine failure; pre-auth approved PA-8842; "
        "not out-of-network",
        "Not out-of-network; NOT roadside only; immobile vehicle; winched onto flatbed; "
        "preauthorization PA-8842 documented",
        "NOT roadside assistance only; stalled on highway; tow truck dispatched; "
        "pre-auth PA-8842 on file; not out of network",
    ]

    query_variants = [
        "Use the REPL to parse the policy and CSV. "
        "If needed, call llm_query to classify the note (emergency towing? preauth present? immobile?). "
        "Pay attention to negations like 'not out-of-network' and 'NOT roadside assistance only'. "
        "Return EXACTLY: eligible=<true|false>, payout=<int>. No extra text. "
        "Question: For claim C100, is it eligible and what is the payout?",
        "Read the policy and CSV via the REPL, then decide eligibility and payout for claim C100. "
        "Watch for negations (e.g., NOT roadside assistance only, not out-of-network). "
        "Return EXACTLY: eligible=<true|false>, payout=<int>.",
    ]

    claims_csv = make_claims_csv(note_variants[0])
    context = [policy, claims_csv]
    query = query_variants[0]

    rows = parse_csv_rows(claims_csv)
    target = next(row for row in rows if row["id"] == "C100")
    expected = compute_expected(target)
    print("Expected:", expected)

    use_scripted = os.environ.get("USE_SCRIPTED_DEMO") == "1"
    if use_scripted:
        weak_root = [
            """```repl
import csv
from datetime import date

policy, claims_csv = context
reader = csv.DictReader([line for line in claims_csv.splitlines() if line.strip()])
rows = list(reader)
row = [r for r in rows if r["id"] == "C100"][0]
amount = int(row["amount"])
service_date = date.fromisoformat(row["service_date"])
filed_date = date.fromisoformat(row["filed_date"])
note = row["note"]
note_lower = note.lower()

if "out of network" in note_lower or "oon" in note_lower:
    eligible = False
elif (filed_date - service_date).days > 30:
    eligible = False
else:
    if row["plan"] == "Gold":
        eligible = True
    elif row["plan"] == "Silver":
        ans = llm_query(
            f"Answer yes or no: does the note imply emergency towing? Note: {note}"
        )
        emergency = ans.strip().lower() == "yes"
        preauth = "pa-" in note_lower or "preauth" in note_lower
        eligible = emergency and preauth
    else:
        eligible = False

payout = 0
if eligible:
    deductible = 200
    cap = 1500
    payout = min(max(amount - deductible, 0), cap)
answer = f"eligible={str(eligible).lower()}, payout={payout}"
print(answer)
```""",
            "FINAL_VAR(answer)",
        ]

        strong_root = [
            """```repl
import csv
import re
from datetime import date

policy, claims_csv = context
reader = csv.DictReader([line for line in claims_csv.splitlines() if line.strip()])
rows = list(reader)
row = [r for r in rows if r["id"] == "C100"][0]
amount = int(row["amount"])
service_date = date.fromisoformat(row["service_date"])
filed_date = date.fromisoformat(row["filed_date"])
note = row["note"]

note_lower = note.lower()
def affirmed(pattern):
    for match in re.finditer(pattern, note_lower):
        prefix = note_lower[: match.start()]
        tokens = re.findall(r"\\b\\w+\\b", prefix)
        last_tokens = tokens[-3:]
        if any(
            token in ("no", "not", "never", "without", "denies", "denied", "declines", "declined")
            for token in last_tokens
        ):
            continue
        return True
    return False

out_of_network = (
    affirmed(r"\\bout of network\\b")
    or affirmed(r"\\bout-of-network\\b")
    or affirmed(r"\\boon\\b")
    or affirmed(r"\\bnon[- ]network\\b")
)
roadside_only = (
    affirmed(r"\\broadside assistance only\\b")
    or affirmed(r"\\broadside only\\b")
    or affirmed(r"\\bjump start only\\b")
    or affirmed(r"\\btire change only\\b")
    or affirmed(r"\\blockout only\\b")
)
immobile = (
    affirmed(r"\\bimmobile\\b")
    or affirmed(r"\\bwould not start\\b")
    or affirmed(r"\\bunable to move\\b")
    or affirmed(r"\\bdisabled vehicle\\b")
    or affirmed(r"\\bstalled on highway\\b")
    or affirmed(r"\\bcannot move\\b")
)

preauth = bool(re.search(r"\\bpa-\\d{3,}\\b", note_lower))
preauth = preauth or "preauth" in note_lower or "pre-auth" in note_lower
preauth = preauth or "pre authorization" in note_lower or "pre-authorization" in note_lower

ans = llm_query(
    "Answer yes or no only: does the note imply emergency towing? "
    f"Note: {note}"
)
ans_norm = ans.strip().lower()
emergency = ans_norm.startswith("y") or "yes" in ans_norm

if out_of_network or roadside_only or (filed_date - service_date).days > 30:
    eligible = False
else:
    if row["plan"] == "Gold":
        eligible = True
    elif row["plan"] == "Silver":
        eligible = emergency and preauth and immobile
    elif row["plan"] == "Bronze":
        eligible = emergency and immobile and amount <= 1200
    else:
        eligible = False

payout = 0
if eligible:
    deductible = 200
    cap = 1500
    payout = min(max(amount - deductible, 0), cap)
answer = f"eligible={str(eligible).lower()}, payout={payout}"
print(answer)
```""",
            "FINAL_VAR(answer)",
        ]

        weak_sub = "Yes, it indicates emergency towing."
        strong_sub = "yes"

        weak_label = "Scripted weak root/sub"
        strong_label = "Scripted strong root/sub"
        weak_answer = run_case(weak_label, weak_root, weak_sub, query, context)
        strong_answer = run_case(strong_label, strong_root, strong_sub, query, context)
    else:
        num_trials = int(os.environ.get("NUM_TRIALS", "1"))
        seed = os.environ.get("RANDOM_SEED")
        rng = random.Random(int(seed)) if seed else random.Random()

        weak_root_model = os.environ.get("WEAK_ROOT_MODEL", "gpt-4.1-nano")
        weak_sub_model = os.environ.get("WEAK_SUB_MODEL", "gpt-4.1-nano")
        strong_root_model = os.environ.get("STRONG_ROOT_MODEL", "gpt-5.2")
        strong_sub_model = os.environ.get("STRONG_SUB_MODEL", "gpt-5.2")
        default_attempts = int(os.environ.get("MAX_ATTEMPTS", "2"))
        weak_attempts = int(os.environ.get("MAX_ATTEMPTS_WEAK", str(default_attempts)))
        strong_attempts = int(os.environ.get("MAX_ATTEMPTS_STRONG", str(default_attempts)))
        verbose_trials = os.environ.get("VERBOSE_TRIALS") == "1"

        weak_label = format_model_pair_label(weak_root_model, weak_sub_model)
        strong_label = format_model_pair_label(strong_root_model, strong_sub_model)
        weak_counts = {"correct": 0, "incorrect": 0, "invalid": 0, "error": 0}
        strong_counts = {"correct": 0, "incorrect": 0, "invalid": 0, "error": 0}

        for idx in range(num_trials):
            note_text = rng.choice(note_variants)
            trial_query = rng.choice(query_variants)
            claims_csv = make_claims_csv(note_text)
            context = [policy, claims_csv]
            rows = parse_csv_rows(claims_csv)
            target = next(row for row in rows if row["id"] == "C100")
            expected = compute_expected(target)

            if verbose_trials or num_trials <= 5:
                print(f"Trial {idx + 1} expected: {expected}")

            weak_answer = run_live_case(
                weak_label,
                weak_root_model,
                weak_sub_model,
                trial_query,
                context,
                expected,
                weak_attempts,
            )
            strong_answer = run_live_case(
                strong_label,
                strong_root_model,
                strong_sub_model,
                trial_query,
                context,
                expected,
                strong_attempts,
            )

            weak_counts[classify_outcome(weak_answer, expected)] += 1
            strong_counts[classify_outcome(strong_answer, expected)] += 1

        def summary(label, counts):
            total = sum(counts.values())
            rate = (counts["correct"] / total * 100) if total else 0
            print(
                f"{label} pass rate: {counts['correct']}/{total} ({rate:.0f}%) | "
                f"incorrect={counts['incorrect']}, invalid={counts['invalid']}, error={counts['error']}"
            )

        summary(weak_label, weak_counts)
        summary(strong_label, strong_counts)

        if (
            weak_counts["correct"] < strong_counts["correct"]
            and strong_counts["correct"] == num_trials
        ):
            print(f"Result: {weak_label} fails, {strong_label} succeeds.")

        return

    if weak_answer != expected and strong_answer == expected:
        print(f"Result: {weak_label} fails, {strong_label} succeeds.")


if __name__ == "__main__":
    main()
