import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_corpus", required=True)
    parser.add_argument("--output_queries", required=True)
    args = parser.parse_args()

    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
    except Exception as exc:
        raise RuntimeError("Install beir to use this tool: pip install beir") from exc

    url = util.download_and_unzip(args.dataset, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=url).load(split="test")

    with open(args.output_corpus, "w", encoding="utf-8") as handle:
        for doc_id, doc in corpus.items():
            handle.write(
                __import__("json").dumps({"doc_id": doc_id, "text": doc.get("text", ""), "title": doc.get("title")})
                + "\n"
            )

    with open(args.output_queries, "w", encoding="utf-8") as handle:
        for qid, query in queries.items():
            gold_doc_ids = list(qrels.get(qid, {}).keys())
            handle.write(
                __import__("json").dumps({"qid": qid, "question": query, "gold_doc_ids": gold_doc_ids})
                + "\n"
            )

    print("Wrote", args.output_corpus, args.output_queries)


if __name__ == "__main__":
    main()
