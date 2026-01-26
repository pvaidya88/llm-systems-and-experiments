from datetime import date
import re

from rlm_demo import RLM, LLMClient


class ScriptedLLM(LLMClient):
    def __init__(self, replies):
        self._replies = list(replies)

    def complete(self, messages):
        if not self._replies:
            raise RuntimeError("ScriptedLLM ran out of replies")
        return self._replies.pop(0)


def to_lines(transactions):
    def sort_key(row):
        return row["date"], row["id"]

    lines = []
    for row in sorted(transactions, key=sort_key):
        lines.append(
            f"{row['id']},{row['customer']},{row['date']},{row['amount']},"
            f"{row['type']},{row['category']}"
        )
    return lines


def chunk_ledger(lines):
    july_aug = []
    sep_and_other = []
    for line in lines:
        parts = line.split(",")
        if len(parts) != 6:
            continue
        date_str = parts[2]
        month = int(date_str.split("-")[1])
        if month in (7, 8):
            july_aug.append(line)
        else:
            sep_and_other.append(line)

    chunk_a = ["# Q3 2024 Ledger (July-August)", "id,customer,date,amount,type,category"]
    chunk_a.extend(july_aug)
    chunk_b = ["# September Ledger", "id,customer,date,amount,type,category"]
    chunk_b.extend(sep_and_other)
    return ["\n".join(chunk_a), "\n".join(chunk_b)]


def retrieve_top_k(query, chunks, k=1):
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))

    def score(chunk):
        chunk_terms = set(re.findall(r"[a-z0-9]+", chunk.lower()))
        return len(chunk_terms & query_terms)

    ranked = sorted(
        enumerate(chunks),
        key=lambda item: (-score(item[1]), item[0]),
    )
    return [chunks[idx] for idx, _ in ranked[:k]]


def parse_transactions(text):
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#") or line.startswith("id,"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        _, customer, date_str, amount_str, tx_type, category = parts
        rows.append(
            {
                "customer": customer,
                "date": date_str,
                "amount": int(amount_str),
                "type": tx_type,
                "category": category,
            }
        )
    return rows


def top_category(transactions, customer, start_date, end_date):
    totals = {}
    for row in transactions:
        if row["customer"] != customer:
            continue
        if row["type"] == "chargeback":
            continue
        d = date.fromisoformat(row["date"])
        if not (start_date <= d <= end_date):
            continue
        totals[row["category"]] = totals.get(row["category"], 0) + row["amount"]
    if not totals:
        return "<no data>"
    cat, amt = max(totals.items(), key=lambda item: item[1])
    return f"{cat} ({amt})"


def main():
    transactions = [
        {"id": "t1", "customer": "C42", "date": "2024-07-03", "amount": 800, "type": "sale", "category": "Books"},
        {"id": "t2", "customer": "C42", "date": "2024-07-12", "amount": 100, "type": "sale", "category": "Electronics"},
        {"id": "t3", "customer": "C99", "date": "2024-07-10", "amount": 500, "type": "sale", "category": "Electronics"},
        {"id": "t4", "customer": "C42", "date": "2024-08-02", "amount": 600, "type": "sale", "category": "Books"},
        {"id": "t5", "customer": "C42", "date": "2024-08-05", "amount": 150, "type": "sale", "category": "Electronics"},
        {"id": "t6", "customer": "C42", "date": "2024-08-18", "amount": 200, "type": "chargeback", "category": "Books"},
        {"id": "t7", "customer": "C42", "date": "2024-09-10", "amount": 2000, "type": "sale", "category": "Electronics"},
        {"id": "t8", "customer": "C42", "date": "2024-09-15", "amount": 900, "type": "sale", "category": "Home"},
        {"id": "t9", "customer": "C42", "date": "2024-09-28", "amount": 300, "type": "chargeback", "category": "Electronics"},
        {"id": "t10", "customer": "C07", "date": "2024-09-01", "amount": 700, "type": "sale", "category": "Home"},
        {"id": "t11", "customer": "C42", "date": "2024-10-02", "amount": 1200, "type": "sale", "category": "Electronics"},
        {"id": "t12", "customer": "C42", "date": "2024-06-20", "amount": 400, "type": "sale", "category": "Books"},
    ]

    query = (
        "For customer C42 in Q3 2024 (Jul 1 - Sep 30), excluding chargebacks, "
        "which category has the highest net amount and what is the net amount?"
    )

    lines = to_lines(transactions)
    chunks = chunk_ledger(lines)
    retrieved = retrieve_top_k(query, chunks, k=1)
    rag_context = "\n".join(retrieved)
    rag_answer = top_category(
        parse_transactions(rag_context),
        customer="C42",
        start_date=date(2024, 7, 1),
        end_date=date(2024, 9, 30),
    )

    replies = [
        """```repl
from datetime import date
totals = {}
for row in context:
    if row["customer"] != "C42":
        continue
    if row["type"] == "chargeback":
        continue
    d = date.fromisoformat(row["date"])
    if not (date(2024, 7, 1) <= d <= date(2024, 9, 30)):
        continue
    totals[row["category"]] = totals.get(row["category"], 0) + row["amount"]
best_cat, best_amt = max(totals.items(), key=lambda item: item[1])
answer = f"{best_cat} ({best_amt})"
print(answer)
```""",
        "FINAL_VAR(answer)",
    ]

    rlm = RLM(ScriptedLLM(replies))
    rlm_answer = rlm.answer(query, transactions)

    expected = top_category(
        transactions,
        customer="C42",
        start_date=date(2024, 7, 1),
        end_date=date(2024, 9, 30),
    )

    print("Query:", query)
    print("RAG retrieved chunk:", retrieved[0].splitlines()[0])
    print("RAG answer:", rag_answer)
    print("RLM answer:", rlm_answer)
    print("Expected:", expected)
    if rag_answer != expected and rlm_answer == expected:
        print("Result: RLM succeeds while RAG fails (retrieval missed part of Q3).")


if __name__ == "__main__":
    main()
