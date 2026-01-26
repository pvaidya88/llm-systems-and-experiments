import os

from rlm_demo import OpenAICompatibleClient, RLM


def main():
    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000")
    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise RuntimeError("Set LLM_MODEL to your model name")

    client = OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=0.2,
    )

    context = """\
Alice has a cat named Snowball.
She also has a dog named Rex.
"""
    query = "What is the cat's name?"

    rlm = RLM(client)
    answer = rlm.answer(query, context)
    print(answer)


if __name__ == "__main__":
    main()