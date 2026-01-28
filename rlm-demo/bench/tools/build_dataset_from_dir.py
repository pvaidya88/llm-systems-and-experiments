import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suffix", default=".txt")
    args = parser.parse_args()

    with open(args.output, "w", encoding="utf-8") as handle:
        for root, _, files in os.walk(args.input):
            for fname in files:
                if args.suffix and not fname.lower().endswith(args.suffix):
                    continue
                path = os.path.join(root, fname)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                doc_id = os.path.relpath(path, args.input)
                handle.write(json.dumps({"doc_id": doc_id, "text": text, "source": path}) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
