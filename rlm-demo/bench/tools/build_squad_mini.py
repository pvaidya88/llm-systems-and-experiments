import argparse
import json

from datasets import load_dataset


def _is_ascii(text: str) -> bool:
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-corpus", required=True)
    parser.add_argument("--out-queries", required=True)
    parser.add_argument("--max-questions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-context-chars", type=int, default=1200)
    args = parser.parse_args()

    dataset = load_dataset("squad", split="train")
    dataset = dataset.shuffle(seed=args.seed)

    doc_map = {}
    corpus = []
    queries = []

    for row in dataset:
        if len(queries) >= args.max_questions:
            break
        context = (row.get("context") or "").strip()
        question = (row.get("question") or "").strip()
        answers = (row.get("answers") or {}).get("text") or []
        if not answers:
            continue
        answer = (answers[0] or "").strip()
        title = (row.get("title") or "unknown").strip()
        if not context or not question or not answer:
            continue
        if len(context) > args.max_context_chars:
            continue
        if not (_is_ascii(context) and _is_ascii(question) and _is_ascii(answer) and _is_ascii(title)):
            continue

        doc_id = doc_map.get(context)
        if doc_id is None:
            doc_id = f"doc{len(doc_map)}"
            doc_map[context] = doc_id
            corpus.append(
                {
                    "doc_id": doc_id,
                    "text": context,
                    "title": title,
                    "source": f"SQuAD v1.1 / Wikipedia / {title}",
                }
            )

        queries.append(
            {
                "qid": f"q{len(queries) + 1}",
                "question": question,
                "answer": answer,
                "gold_doc_ids": [doc_id],
            }
        )

    with open(args.out_corpus, "w", encoding="utf-8") as corpus_handle:
        for item in corpus:
            corpus_handle.write(json.dumps(item, ensure_ascii=True) + "\n")

    with open(args.out_queries, "w", encoding="utf-8") as queries_handle:
        for item in queries:
            queries_handle.write(json.dumps(item, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
