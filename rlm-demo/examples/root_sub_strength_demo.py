import csv
import io
import os
import random
import re
import unicodedata
from datetime import date

from rlm_demo import LLMClient, OpenAIResponsesClient, RLM, RLMOptions

OUTPUT_PATTERN = re.compile(
    r"^eligible\s*=\s*(true|false)\s*,\s*payout\s*=\s*(\d+)\s*$",
    re.IGNORECASE,
)

YESNO_QUESTIONS = [
    ("emergency", "Does the note imply emergency towing?"),
    ("preauth", "Is a pre-authorization present?"),
    ("immobile", "Was the vehicle immobile?"),
    ("out_of_network", "Is it out-of-network?"),
    ("roadside_only", "Is it roadside assistance only?"),
]


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


def strip_invisible(text):
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("C"))


def normalize_answer(text):
    if text is None:
        return None
    cleaned = strip_invisible(text.strip())
    match = OUTPUT_PATTERN.fullmatch(cleaned)
    if not match:
        return None
    return f"eligible={match.group(1).lower()}, payout={int(match.group(2))}"


def normalize_yesno(text):
    if text is None:
        return None
    cleaned = strip_invisible(text.strip()).lower()
    if cleaned.startswith("y"):
        return "yes"
    if cleaned.startswith("n"):
        return "no"
    return None


def redact_note_in_csv(csv_text):
    reader = csv.DictReader([line for line in csv_text.splitlines() if line.strip()])
    rows = []
    for row in reader:
        row = {key.strip(): value.strip() for key, value in row.items()}
        if "note" in row:
            row["note"] = "<redacted>"
        rows.append(row)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().strip()


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


def build_strict_sub_repl_query(hidden_note, vote_k=1):
    helper = ""
    if vote_k > 1:
        helper = f"""
def majority_yesno(fn, question):
    votes = []
    for _ in range({vote_k}):
        votes.append(fn(question))
    yes_votes = sum(1 for v in votes if v == "yes")
    no_votes = sum(1 for v in votes if v == "no")
    return "yes" if yes_votes >= no_votes else "no"
"""
    fn_name = "note_yesno" if hidden_note else "llm_query_yesno"
    call = "majority_yesno" if vote_k > 1 else fn_name
    note_line = "" if hidden_note else "note = row[\"note\"]\n\n"
    def q(text):
        if hidden_note:
            return f"\"{text}\""
        return f"\"{text} Note: \" + note"
    repl_block = f"""```repl
import csv
from datetime import date
policy, claims_csv = context
reader = csv.DictReader([line for line in claims_csv.splitlines() if line.strip()])
rows = list(reader)
row = [r for r in rows if r["id"] == "C100"][0]
amount = int(row["amount"])
service_date = date.fromisoformat(row["service_date"])
filed_date = date.fromisoformat(row["filed_date"])
{note_line}{helper}
emergency = {call}({fn_name}, {q("Does the note imply emergency towing?")}) == "yes"
preauth = {call}({fn_name}, {q("Is a pre-authorization present?")}) == "yes"
immobile = {call}({fn_name}, {q("Was the vehicle immobile?")}) == "yes"
out_of_network = {call}({fn_name}, {q("Is it out-of-network?")}) == "yes"
roadside_only = {call}({fn_name}, {q("Is it roadside assistance only?")}) == "yes"

timely = (filed_date - service_date).days <= 30

if out_of_network or roadside_only or not timely:
    eligible = False
elif row["plan"] == "Gold":
    eligible = True
elif row["plan"] == "Silver":
    eligible = emergency and preauth and immobile
elif row["plan"] == "Bronze":
    eligible = emergency and immobile and amount <= 1200
else:
    eligible = False

payout = 0
if eligible:
    deductibles = {{"Gold": 100, "Silver": 200, "Bronze": 300}}
    caps = {{"Gold": 3000, "Silver": 1500, "Bronze": 800}}
    payout = min(max(amount - deductibles.get(row["plan"], 0), 0), caps.get(row["plan"], amount))

answer = f"eligible={{str(eligible).lower()}}, payout={{payout}}"
```"""
    return (
        "In your next reply, output EXACTLY the following REPL block, unchanged, "
        "then on a new line output FINAL_VAR(answer). Do not add any other text.\n\n"
        + repl_block
    )


def build_strict_repl_query():
    return build_strict_sub_repl_query(hidden_note=False, vote_k=1)


def build_strict_repl_query_hidden(vote_k=1):
    return build_strict_sub_repl_query(hidden_note=True, vote_k=vote_k)


def build_oracle_flags_query(flags):
    emergency = "True" if flags.get("emergency") else "False"
    preauth = "True" if flags.get("preauth") else "False"
    immobile = "True" if flags.get("immobile") else "False"
    out_of_network = "True" if flags.get("out_of_network") else "False"
    roadside_only = "True" if flags.get("roadside_only") else "False"
    repl_block = f"""```repl
import csv
from datetime import date

policy, claims_csv = context
reader = csv.DictReader([line for line in claims_csv.splitlines() if line.strip()])
rows = list(reader)
row = [r for r in rows if r["id"] == "C100"][0]
amount = int(row["amount"])
service_date = date.fromisoformat(row["service_date"])
filed_date = date.fromisoformat(row["filed_date"])

emergency = {emergency}
preauth = {preauth}
immobile = {immobile}
out_of_network = {out_of_network}
roadside_only = {roadside_only}

timely = (filed_date - service_date).days <= 30

if out_of_network or roadside_only or not timely:
    eligible = False
elif row["plan"] == "Gold":
    eligible = True
elif row["plan"] == "Silver":
    eligible = emergency and preauth and immobile
elif row["plan"] == "Bronze":
    eligible = emergency and immobile and amount <= 1200
else:
    eligible = False

payout = 0
if eligible:
    deductibles = {{"Gold": 100, "Silver": 200, "Bronze": 300}}
    caps = {{"Gold": 3000, "Silver": 1500, "Bronze": 800}}
    payout = min(max(amount - deductibles.get(row["plan"], 0), 0), caps.get(row["plan"], amount))

answer = f"eligible={{str(eligible).lower()}}, payout={{payout}}"
```"""
    return (
        "In your next reply, output EXACTLY the following REPL block, unchanged, "
        "then on a new line output FINAL_VAR(answer). Do not add any other text.\n\n"
        + repl_block
    )

def build_strict_root_repl_query():
    repl_block = """```repl
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

emergency = (
    affirmed(r"\\bemergency towing\\b")
    or affirmed(r"\\btow(?:ed|ing)?\\b")
    or affirmed(r"\\btow truck\\b")
    or affirmed(r"\\bflatbed\\b")
    or affirmed(r"\\bwrecker\\b")
    or affirmed(r"\\bwinch(?:ed|ing)?\\b")
    or affirmed(r"\\brecovery vehicle\\b")
    or affirmed(r"\\bvehicle recovery\\b")
    or affirmed(r"\\bpulled from ditch\\b")
)
immobile = (
    affirmed(r"\\bimmobile\\b")
    or affirmed(r"\\bwould not start\\b")
    or affirmed(r"\\bunable to move\\b")
    or affirmed(r"\\bdisabled vehicle\\b")
    or affirmed(r"\\bstalled on highway\\b")
    or affirmed(r"\\bcannot move\\b")
)
roadside_only = (
    affirmed(r"\\broadside assistance only\\b")
    or affirmed(r"\\broadside only\\b")
    or affirmed(r"\\bjump start only\\b")
    or affirmed(r"\\btire change only\\b")
    or affirmed(r"\\blockout only\\b")
)
out_of_network = (
    affirmed(r"\\bout of network\\b")
    or affirmed(r"\\bout-of-network\\b")
    or affirmed(r"\\boon\\b")
    or affirmed(r"\\bnon[- ]network\\b")
)

preauth = bool(re.search(r"\\bpa-\\d{3,}\\b", note_lower))
preauth = preauth or "preauth" in note_lower or "pre-auth" in note_lower
preauth = preauth or "pre authorization" in note_lower or "pre-authorization" in note_lower

timely = (filed_date - service_date).days <= 30

if out_of_network or roadside_only or not timely:
    eligible = False
elif row["plan"] == "Gold":
    eligible = True
elif row["plan"] == "Silver":
    eligible = emergency and preauth and immobile
elif row["plan"] == "Bronze":
    eligible = emergency and immobile and amount <= 1200
else:
    eligible = False

payout = 0
if eligible:
    deductibles = {"Gold": 100, "Silver": 200, "Bronze": 300}
    caps = {"Gold": 3000, "Silver": 1500, "Bronze": 800}
    payout = min(max(amount - deductibles.get(row["plan"], 0), 0), caps.get(row["plan"], amount))

answer = f"eligible={str(eligible).lower()}, payout={payout}"
```"""
    return (
        "In your next reply, output EXACTLY the following REPL block, unchanged, "
        "then on a new line output FINAL_VAR(answer). Do not add any other text.\n\n"
        + repl_block
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


def flags_to_dict(emergency, preauth, out_of_network, immobile, roadside_only):
    return {
        "emergency": emergency,
        "preauth": preauth,
        "out_of_network": out_of_network,
        "immobile": immobile,
        "roadside_only": roadside_only,
    }


def compute_expected_from_flags(row, flags):
    plan = row["plan"]
    amount = int(row["amount"])
    service_date = date.fromisoformat(row["service_date"])
    filed_date = date.fromisoformat(row["filed_date"])

    emergency = bool(flags.get("emergency"))
    preauth = bool(flags.get("preauth"))
    out_of_network = bool(flags.get("out_of_network"))
    immobile = bool(flags.get("immobile"))
    roadside_only = bool(flags.get("roadside_only"))
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


_CLIENT_CACHE = {}


def normalize_text_verbosity(model_name, value):
    if not value:
        return None
    name = (model_name or "").strip().lower()
    if name.startswith("gpt-4.1-nano"):
        return "medium"
    return value


def get_openai_client(model_name, effort, text_verbosity, api_key, base_url):
    key = (model_name, effort, text_verbosity, base_url)
    if key in _CLIENT_CACHE:
        return _CLIENT_CACHE[key]
    client = OpenAIResponsesClient(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        reasoning_effort=effort if supports_reasoning_model(model_name) else None,
        text_verbosity=normalize_text_verbosity(model_name, text_verbosity),
    )
    _CLIENT_CACHE[key] = client
    return client


def ask_yesno_with_client(client, question, note, max_retries):
    prompt = f"{question} Note: {note}"
    last_response = ""
    for attempt in range(max(1, max_retries)):
        if attempt == 0:
            response = client.complete([{"role": "user", "content": prompt}])
        else:
            response = client.complete(
                [{"role": "user", "content": f"Answer yes or no only. {prompt}"}]
            )
        last_response = response or ""
        normalized = normalize_yesno(last_response)
        if normalized is not None:
            return normalized
    return None


def get_sub_flags(client, note, max_retries):
    flags = {}
    for key, question in YESNO_QUESTIONS:
        answer = ask_yesno_with_client(client, question, note, max_retries)
        if answer is None:
            return None
        flags[key] = answer == "yes"
    return flags


def run_case(
    label,
    root_replies,
    sub_reply,
    query,
    context,
    *,
    min_sub_calls=0,
    include_cost_hint=True,
    max_steps=10,
    max_sub_calls=None,
    hidden_note=None,
):
    log_repl = os.environ.get("LOG_REPL_OUTPUTS") == "1"
    options = RLMOptions(
        require_repl=True,
        retry_on_invalid=True,
        log_repl_outputs=log_repl,
        max_steps=max_steps,
        min_sub_calls=min_sub_calls,
        include_cost_hint=include_cost_hint,
        hidden_note=hidden_note,
    )
    if max_sub_calls is not None:
        options.max_sub_calls = max_sub_calls
    rlm = RLM(
        root_llm=ScriptedLLM(root_replies),
        sub_llm=StaticLLM(sub_reply),
        options=options,
    )
    answer = rlm.answer(query, context)
    normalized = normalize_answer(answer) or answer
    print(f"{label} answer: {normalized}")
    return normalized


def run_live_case(
    label,
    root_model,
    sub_model,
    query,
    context,
    expected,
    max_attempts,
    *,
    min_sub_calls=0,
    include_cost_hint=True,
    max_steps=10,
    max_sub_calls=None,
    hidden_note=None,
):
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

    root_client = get_openai_client(
        root_model, root_effort, text_verbosity, api_key, base_url
    )
    sub_client = get_openai_client(sub_model, sub_effort, text_verbosity, api_key, base_url)
    log_repl = os.environ.get("LOG_REPL_OUTPUTS") == "1"
    options = RLMOptions(
        require_repl=True,
        retry_on_invalid=True,
        log_repl_outputs=log_repl,
        max_steps=max_steps,
        min_sub_calls=min_sub_calls,
        include_cost_hint=include_cost_hint,
        hidden_note=hidden_note,
    )
    if max_sub_calls is not None:
        options.max_sub_calls = max_sub_calls
    rlm = RLM(
        root_llm=root_client,
        sub_llm=sub_client,
        options=options,
    )

    last_error = None
    last_invalid = None
    last_normalized = None
    last_incorrect = False
    for attempt in range(max_attempts):
        attempt_query = build_query(query, attempt, last_error, incorrect=last_incorrect)
        try:
            answer = rlm.answer(attempt_query, context)
        except Exception as exc:
            print(f"{label} error: {exc}")
            return f"<error: {exc}>"
        normalized = normalize_answer(answer)
        if normalized:
            if normalized == expected:
                print(f"{label} answer: {normalized}")
                return normalized
            last_normalized = normalized
            last_error = f"incorrect answer: {normalized}"
            last_incorrect = True
            continue
        last_invalid = answer
        last_error = f"invalid format: {answer}"
        last_incorrect = False

    if last_normalized is not None:
        print(f"{label} incorrect output: {last_normalized}")
        return last_normalized

    print(f"{label} invalid output: {last_invalid}")
    if os.environ.get("LOG_INVALID_REPR") == "1":
        print(f"{label} invalid repr: {last_invalid!r}")
    return f"<invalid: {last_invalid}>"


def select_query(
    force_sub_lm,
    force_root_lm,
    strict_template,
    root_strict_template,
    hide_note_from_root,
    sub_vote_k,
    query_variants,
    sub_lm_query_variants,
    sub_lm_query_variants_hidden,
    root_lm_query_variants,
    rng,
    randomize,
):
    if strict_template and not force_root_lm:
        return build_strict_sub_repl_query(
            hidden_note=hide_note_from_root, vote_k=sub_vote_k
        )
    if force_root_lm:
        if root_strict_template:
            return build_strict_root_repl_query()
        return rng.choice(root_lm_query_variants) if randomize else root_lm_query_variants[0]
    if force_sub_lm:
        variants = sub_lm_query_variants_hidden if hide_note_from_root else sub_lm_query_variants
        return rng.choice(variants) if randomize else variants[0]
    return rng.choice(query_variants) if randomize else query_variants[0]


def build_trial_set(
    num_trials,
    rng,
    fixed_trials,
    note_variants,
    policy,
    force_sub_lm,
    force_root_lm,
    strict_template,
    root_strict_template,
    hide_note_from_root,
    sub_vote_k,
    query_variants,
    sub_lm_query_variants,
    sub_lm_query_variants_hidden,
    root_lm_query_variants,
):
    trials = []
    for idx in range(num_trials):
        note_text = note_variants[0] if fixed_trials else rng.choice(note_variants)
        claims_csv = make_claims_csv(note_text)
        context_csv = redact_note_in_csv(claims_csv) if hide_note_from_root else claims_csv
        context = [policy, context_csv]
        rows = parse_csv_rows(claims_csv)
        row = next(r for r in rows if r["id"] == "C100")
        expected = compute_expected(row)
        oracle_flags = flags_to_dict(*parse_note_flags(note_text))
        trial_query = select_query(
            force_sub_lm,
            force_root_lm,
            strict_template,
            root_strict_template,
            hide_note_from_root,
            sub_vote_k,
            query_variants,
            sub_lm_query_variants,
            sub_lm_query_variants_hidden,
            root_lm_query_variants,
            rng,
            randomize=not fixed_trials,
        )
        trials.append(
            {
                "index": idx + 1,
                "note_text": note_text,
                "context": context,
                "expected": expected,
                "row": row,
                "oracle_flags": oracle_flags,
                "query": trial_query,
            }
        )
    return trials


def summarize_counts(label, counts):
    total = sum(counts.values())
    rate = (counts["correct"] / total * 100) if total else 0
    print(
        f"{label} pass rate: {counts['correct']}/{total} ({rate:.0f}%) | "
        f"incorrect={counts['incorrect']}, invalid={counts['invalid']}, error={counts['error']}"
    )


def run_oracle_ablations(
    trials,
    root_models,
    sub_models,
    api_key,
    base_url,
    text_verbosity,
    root_effort,
    sub_effort,
    max_attempts,
):
    yesno_retries = int(os.environ.get("LLM_YESNO_MAX_RETRIES", "4"))

    oracle_root_counts = {model: {"correct": 0, "incorrect": 0, "invalid": 0} for model in sub_models}
    for sub_model in sub_models:
        sub_client = get_openai_client(
            sub_model, sub_effort, text_verbosity, api_key, base_url
        )
        for trial in trials:
            flags = get_sub_flags(sub_client, trial["note_text"], yesno_retries)
            if flags is None:
                oracle_root_counts[sub_model]["invalid"] += 1
                continue
            predicted = compute_expected_from_flags(trial["row"], flags)
            if predicted == trial["expected"]:
                oracle_root_counts[sub_model]["correct"] += 1
            else:
                oracle_root_counts[sub_model]["incorrect"] += 1

    oracle_sub_counts = {model: {"correct": 0, "incorrect": 0, "invalid": 0, "error": 0} for model in root_models}
    for root_model in root_models:
        label = f"oracle_sub root={root_model}"
        for trial in trials:
            oracle_query = build_oracle_flags_query(trial["oracle_flags"])
            answer = run_live_case(
                label,
                root_model,
                root_model,
                oracle_query,
                trial["context"],
                trial["expected"],
                max_attempts,
                min_sub_calls=0,
                include_cost_hint=False,
                max_steps=int(os.environ.get("MAX_STEPS", "10")),
                max_sub_calls=0,
                hidden_note=None,
            )
            oracle_sub_counts[root_model][classify_outcome(answer, trial["expected"])] += 1

    for sub_model, counts in oracle_root_counts.items():
        total = sum(counts.values())
        rate = (counts["correct"] / total * 100) if total else 0
        print(
            f"oracle_root sub={sub_model} pass rate: {counts['correct']}/{total} ({rate:.0f}%) | "
            f"incorrect={counts['incorrect']}, invalid={counts['invalid']}"
        )

    for root_model, counts in oracle_sub_counts.items():
        summarize_counts(f"oracle_sub root={root_model}", counts)


def run_full_factorial_suite(
    policy,
    note_variants,
    query_variants,
    sub_lm_query_variants,
    sub_lm_query_variants_hidden,
    root_lm_query_variants,
    weak_root_model,
    strong_root_model,
    weak_sub_model,
    strong_sub_model,
    num_trials,
    rng,
    fixed_trials,
    strict_template,
    root_strict_template,
    hide_note_from_root,
    sub_vote_k,
    max_steps,
    max_attempts,
):
    regimes = [
        ("sub_load_bearing", True, False),
        ("root_load_bearing", False, True),
    ]
    for regime_label, force_sub_lm, force_root_lm in regimes:
        print(f"=== Full factorial: {regime_label} ===")
        trials = build_trial_set(
            num_trials,
            rng,
            fixed_trials,
            note_variants,
            policy,
            force_sub_lm,
            force_root_lm,
            strict_template,
            root_strict_template,
            hide_note_from_root if force_sub_lm else False,
            sub_vote_k,
            query_variants,
            sub_lm_query_variants,
            sub_lm_query_variants_hidden,
            root_lm_query_variants,
        )

        combos = [
            (weak_root_model, weak_sub_model),
            (weak_root_model, strong_sub_model),
            (strong_root_model, weak_sub_model),
            (strong_root_model, strong_sub_model),
        ]
        counts_by_combo = {}
        min_sub_calls = 5 if force_sub_lm else 0
        max_sub_calls = 0 if force_root_lm else None
        if not force_root_lm:
            override = os.environ.get("MAX_SUB_CALLS")
            if override:
                max_sub_calls = int(override)
            elif force_sub_lm and sub_vote_k > 1:
                yesno_retries = int(os.environ.get("LLM_YESNO_MAX_RETRIES", "4"))
                max_sub_calls = max(
                    32, sub_vote_k * len(YESNO_QUESTIONS) * yesno_retries
                )
        include_cost_hint = False

        for root_model, sub_model in combos:
            label = format_model_pair_label(root_model, sub_model)
            counts_by_combo[label] = {"correct": 0, "incorrect": 0, "invalid": 0, "error": 0}
            for trial in trials:
                answer = run_live_case(
                    label,
                    root_model,
                    sub_model,
                    trial["query"],
                    trial["context"],
                    trial["expected"],
                    max_attempts,
                    min_sub_calls=min_sub_calls,
                    include_cost_hint=include_cost_hint,
                    max_steps=max_steps,
                    max_sub_calls=max_sub_calls,
                    hidden_note=trial["note_text"] if (force_sub_lm and hide_note_from_root) else None,
                )
                counts_by_combo[label][classify_outcome(answer, trial["expected"])] += 1

        for label, counts in counts_by_combo.items():
            summarize_counts(label, counts)

        if os.environ.get("ORACLE_ABLATIONS") == "1":
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
            text_verbosity = os.environ.get("OPENAI_TEXT_VERBOSITY", "low")
            root_effort = (os.environ.get("ROOT_MODEL_EFFORT") or "").strip() or None
            sub_effort = (os.environ.get("SUB_MODEL_EFFORT") or "").strip() or None
            run_oracle_ablations(
                trials,
                [weak_root_model, strong_root_model],
                [weak_sub_model, strong_sub_model],
                api_key,
                base_url,
                text_verbosity,
                root_effort,
                sub_effort,
                max_attempts,
            )



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
    note_variants_hard = [
        "NOT roadside assistance only; vehicle immobile; tow truck dispatched; "
        "not out-of-network; no approval code on file",
        "NOT roadside assistance only; tow truck dispatched; vehicle was NOT immobile and could be moved; "
        "pre-auth PA-2222; not out-of-network",
        "Out of network; tow truck dispatched; vehicle immobile; pre-auth PA-1111; "
        "NOT roadside assistance only",
        "Roadside assistance only; jump start only; no tow; pre-auth PA-3333; not out-of-network",
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
    sub_lm_query_variants = [
        "You MUST use the REPL and call llm_query_yesno separately for each yes/no question about the note: "
        "(1) does it imply emergency towing? (2) is a pre-authorization present? "
        "(3) was the vehicle immobile? (4) is it out-of-network? (5) is it roadside assistance only? "
        "Ask each question separately and accept only yes/no. Do NOT infer these flags yourself; "
        "treat llm_query_yesno answers as authoritative. Then compute eligibility and payout from the policy/CSV. "
        "If you do not call llm_query_yesno at least 5 times, your answer will be treated as invalid. "
        "Return EXACTLY: eligible=<true|false>, payout=<int>.",
        "Use the REPL. For the note, call llm_query_yesno five times (emergency towing, preauth present, immobile, "
        "out-of-network, roadside-only). Use ONLY those five answers plus the policy/CSV to compute eligibility "
        "and payout. Do not guess or parse the note yourself. Return EXACTLY: eligible=<true|false>, payout=<int>.",
    ]
    sub_lm_query_variants_mitigate = [
        "You MUST use the REPL and call llm_query_yesno for each of these five questions: "
        "emergency towing, pre-auth present, immobile, out-of-network, roadside-only. "
        "To mitigate noise, query each question multiple times and majority-vote the yes/no result. "
        "Do NOT infer the flags yourself; treat the aggregated llm_query_yesno answers as authoritative. "
        "Then compute eligibility and payout. Return EXACTLY: eligible=<true|false>, payout=<int>.",
        "Use the REPL. For each of the five note questions, ask llm_query_yesno multiple times and "
        "take a majority vote. Use ONLY those aggregated answers plus the policy/CSV to compute eligibility "
        "and payout. Do not guess or parse the note yourself. Return EXACTLY: eligible=<true|false>, payout=<int>.",
    ]
    sub_lm_query_variants_hidden = [
        "You MUST use the REPL and call note_yesno separately for each yes/no question about the note: "
        "(1) does it imply emergency towing? (2) is a pre-authorization present? "
        "(3) was the vehicle immobile? (4) is it out-of-network? (5) is it roadside assistance only? "
        "Ask each question separately and accept only yes/no. Do NOT infer these flags yourself; "
        "treat note_yesno answers as authoritative. Then compute eligibility and payout from the policy/CSV. "
        "If you do not call note_yesno at least 5 times, your answer will be treated as invalid. "
        "Return EXACTLY: eligible=<true|false>, payout=<int>.",
        "Use the REPL. For the note, call note_yesno five times (emergency towing, preauth present, immobile, "
        "out-of-network, roadside-only). Use ONLY those five answers plus the policy/CSV to compute eligibility "
        "and payout. Do not guess or parse the note yourself. Return EXACTLY: eligible=<true|false>, payout=<int>.",
    ]
    sub_lm_query_variants_hidden_mitigate = [
        "You MUST use the REPL and call note_yesno for each of the five questions (emergency, preauth, immobile, "
        "out-of-network, roadside-only). Ask each question multiple times and majority-vote the result. "
        "Do NOT infer flags yourself; treat the aggregated note_yesno answers as authoritative. "
        "Then compute eligibility and payout. Return EXACTLY: eligible=<true|false>, payout=<int>.",
        "Use the REPL. For each of the five note questions, call note_yesno multiple times and "
        "take a majority vote. Use ONLY those aggregated answers plus the policy/CSV to compute eligibility "
        "and payout. Do not guess or parse the note yourself. Return EXACTLY: eligible=<true|false>, payout=<int>.",
    ]
    root_lm_query_variants = [
        "You MUST use the REPL but you are NOT allowed to call llm_query or llm_query_yesno. "
        "Parse the note yourself (including negations), compute eligibility and payout from the policy/CSV, "
        "and return EXACTLY: eligible=<true|false>, payout=<int>. No extra text. "
        "If you call llm_query or llm_query_yesno, your answer will be treated as invalid.",
        "Use the REPL to read policy/CSV, but do NOT call llm_query or llm_query_yesno. "
        "You must parse the note yourself (handle negations) and compute eligibility/payout. "
        "Return EXACTLY: eligible=<true|false>, payout=<int>.",
    ]
    sub_variants = sub_lm_query_variants_mitigate if sub_mitigate else sub_lm_query_variants
    sub_hidden_variants = (
        sub_lm_query_variants_hidden_mitigate
        if sub_mitigate
        else sub_lm_query_variants_hidden
    )

    force_sub_lm = os.environ.get("SUBLM_LOAD_BEARING", "1") == "1"
    force_root_lm = os.environ.get("ROOTLM_LOAD_BEARING", "0") == "1"
    full_factorial = os.environ.get("FULL_FACTORIAL") == "1"
    oracle_ablations = os.environ.get("ORACLE_ABLATIONS") == "1"
    sub_mitigate = os.environ.get("SUB_MITIGATE") == "1"
    hard_notes = os.environ.get("HARD_NOTES") == "1"
    sub_vote_k = int(os.environ.get("SUB_VOTE_K", "3" if sub_mitigate else "1"))
    if sub_vote_k < 1:
        sub_vote_k = 1
    root_strict_env = os.environ.get("ROOT_STRICT_REPL_TEMPLATE")
    if root_strict_env is None:
        root_strict_template = force_root_lm
    else:
        root_strict_template = root_strict_env == "1"
    if force_sub_lm and force_root_lm:
        raise RuntimeError("Set only one of SUBLM_LOAD_BEARING or ROOTLM_LOAD_BEARING.")
    hide_note_env = os.environ.get("HIDE_NOTE_FROM_ROOT")
    if hide_note_env is None:
        hide_note_from_root = force_sub_lm
    else:
        hide_note_from_root = hide_note_env == "1"
    if force_root_lm:
        hide_note_from_root = False
    fixed_trials_env = os.environ.get("FIXED_TRIALS")
    if fixed_trials_env is None:
        fixed_trials = force_sub_lm or force_root_lm
    else:
        fixed_trials = fixed_trials_env == "1"
    strict_template_env = os.environ.get("STRICT_REPL_TEMPLATE")
    if strict_template_env is None:
        strict_template = force_sub_lm
    else:
        strict_template = strict_template_env == "1"
    min_sub_calls = int(os.environ.get("MIN_SUB_CALLS", "5" if force_sub_lm else "0"))
    min_sub_calls = max(0, min_sub_calls)
    max_sub_calls = 0 if force_root_lm else None
    if not force_root_lm:
        override = os.environ.get("MAX_SUB_CALLS")
        if override:
            max_sub_calls = int(override)
        elif force_sub_lm and sub_vote_k > 1:
            yesno_retries = int(os.environ.get("LLM_YESNO_MAX_RETRIES", "4"))
            max_sub_calls = max(
                32, sub_vote_k * len(YESNO_QUESTIONS) * yesno_retries
            )
    include_cost_hint = not (force_sub_lm or force_root_lm)
    max_steps = int(os.environ.get("MAX_STEPS", "10"))
    if strict_template and not force_root_lm:
        query = build_strict_sub_repl_query(
            hidden_note=hide_note_from_root, vote_k=sub_vote_k
        )
    else:
        if force_root_lm:
            query = build_strict_root_repl_query() if root_strict_template else root_lm_query_variants[0]
        else:
            if force_sub_lm:
                variants = sub_hidden_variants if hide_note_from_root else sub_variants
                query = variants[0]
            else:
                query = query_variants[0]

    note_variants = note_variants_hard if hard_notes else note_variants
    claims_csv = make_claims_csv(note_variants[0])
    context_csv = redact_note_in_csv(claims_csv) if hide_note_from_root else claims_csv
    context = [policy, context_csv]

    rows = parse_csv_rows(claims_csv)
    target = next(row for row in rows if row["id"] == "C100")
    expected = compute_expected(target)
    print("Expected:", expected)

    if full_factorial:
        num_trials = int(os.environ.get("NUM_TRIALS", "1"))
        seed = os.environ.get("RANDOM_SEED")
        rng = random.Random(int(seed)) if seed else random.Random()
        weak_root_model = os.environ.get("WEAK_ROOT_MODEL", "gpt-4.1-nano")
        weak_sub_model = os.environ.get("WEAK_SUB_MODEL", "gpt-4.1-nano")
        strong_root_model = os.environ.get("STRONG_ROOT_MODEL", "gpt-5.2")
        strong_sub_model = os.environ.get("STRONG_SUB_MODEL", "gpt-5.2")
        default_attempts = int(os.environ.get("MAX_ATTEMPTS", "2"))
        max_attempts = int(os.environ.get("MAX_ATTEMPTS_STRONG", str(default_attempts)))
        run_full_factorial_suite(
            policy,
            note_variants,
            query_variants,
            sub_variants,
            sub_hidden_variants,
            root_lm_query_variants,
            weak_root_model,
            strong_root_model,
            weak_sub_model,
            strong_sub_model,
            num_trials,
            rng,
            fixed_trials,
            strict_template,
            root_strict_template,
            hide_note_from_root,
            sub_vote_k,
            max_steps,
            max_attempts,
        )
        return

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
        weak_answer = run_case(
            weak_label,
            weak_root,
            weak_sub,
            query,
            context,
            min_sub_calls=0,
            include_cost_hint=include_cost_hint,
            max_steps=max_steps,
            max_sub_calls=max_sub_calls,
            hidden_note=note_variants[0] if hide_note_from_root else None,
        )
        strong_answer = run_case(
            strong_label,
            strong_root,
            strong_sub,
            query,
            context,
            min_sub_calls=0,
            include_cost_hint=include_cost_hint,
            max_steps=max_steps,
            max_sub_calls=max_sub_calls,
            hidden_note=note_variants[0] if hide_note_from_root else None,
        )
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
            if fixed_trials:
                note_text = note_variants[0]
                if strict_template and not force_root_lm:
                    trial_query = build_strict_repl_query()
                else:
                    if force_root_lm:
                        trial_query = (
                            build_strict_root_repl_query()
                            if root_strict_template
                            else root_lm_query_variants[0]
                        )
                    else:
                    if force_sub_lm:
                        variants = sub_hidden_variants if hide_note_from_root else sub_variants
                        trial_query = variants[0]
                    else:
                        trial_query = query_variants[0]
            else:
                note_text = rng.choice(note_variants)
                if strict_template and not force_root_lm:
                    trial_query = build_strict_sub_repl_query(
                        hidden_note=hide_note_from_root, vote_k=sub_vote_k
                    )
                else:
                    if force_root_lm:
                        trial_query = (
                            build_strict_root_repl_query()
                            if root_strict_template
                            else rng.choice(root_lm_query_variants)
                        )
                    else:
                        if force_sub_lm:
                            variants = sub_hidden_variants if hide_note_from_root else sub_variants
                            trial_query = rng.choice(variants)
                        else:
                            trial_query = rng.choice(query_variants)
            claims_csv = make_claims_csv(note_text)
            context_csv = redact_note_in_csv(claims_csv) if hide_note_from_root else claims_csv
            context = [policy, context_csv]
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
                min_sub_calls=min_sub_calls,
                include_cost_hint=include_cost_hint,
                max_steps=max_steps,
                max_sub_calls=max_sub_calls,
                hidden_note=note_text if hide_note_from_root else None,
            )
            strong_answer = run_live_case(
                strong_label,
                strong_root_model,
                strong_sub_model,
                trial_query,
                context,
                expected,
                strong_attempts,
                min_sub_calls=min_sub_calls,
                include_cost_hint=include_cost_hint,
                max_steps=max_steps,
                max_sub_calls=max_sub_calls,
                hidden_note=note_text if hide_note_from_root else None,
            )

            weak_counts[classify_outcome(weak_answer, expected)] += 1
            strong_counts[classify_outcome(strong_answer, expected)] += 1

        summarize_counts(weak_label, weak_counts)
        summarize_counts(strong_label, strong_counts)

        if (
            weak_counts["correct"] < strong_counts["correct"]
            and strong_counts["correct"] == num_trials
        ):
            print(f"Result: {weak_label} fails, {strong_label} succeeds.")

        if oracle_ablations:
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
            text_verbosity = os.environ.get("OPENAI_TEXT_VERBOSITY", "low")
            root_effort = (os.environ.get("ROOT_MODEL_EFFORT") or "").strip() or None
            sub_effort = (os.environ.get("SUB_MODEL_EFFORT") or "").strip() or None
            trials = build_trial_set(
                num_trials,
                rng,
                fixed_trials,
                note_variants,
                policy,
                force_sub_lm,
                force_root_lm,
                strict_template,
                root_strict_template,
                hide_note_from_root,
                sub_vote_k,
                query_variants,
                sub_variants,
                sub_hidden_variants,
                root_lm_query_variants,
            )
            run_oracle_ablations(
                trials,
                [weak_root_model, strong_root_model],
                [weak_sub_model, strong_sub_model],
                api_key,
                base_url,
                text_verbosity,
                root_effort,
                sub_effort,
                max(default_attempts, strong_attempts),
            )

        return

    if weak_answer != expected and strong_answer == expected:
        print(f"Result: {weak_label} fails, {strong_label} succeeds.")


if __name__ == "__main__":
    main()
