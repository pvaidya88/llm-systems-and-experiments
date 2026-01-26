import os

from rlm_demo import OpenAIResponsesClient, RLM


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL")
    if not model:
        raise RuntimeError("Set OPENAI_MODEL to your model name")

    text_verbosity = os.environ.get("OPENAI_TEXT_VERBOSITY", "medium")
    root_effort = os.environ.get("OPENAI_REASONING_EFFORT", "xhigh")
    sub_effort = os.environ.get("OPENAI_SUB_REASONING_EFFORT", "low")

    root_client = OpenAIResponsesClient(
        api_key=api_key,
        model=model,
        reasoning_effort=root_effort,
        text_verbosity=text_verbosity,
        store=True,
        include=["reasoning.encrypted_content", "web_search_call.action.sources"],
        tools=[],
    )
    sub_client = OpenAIResponsesClient(
        api_key=api_key,
        model=model,
        reasoning_effort=sub_effort,
        text_verbosity=text_verbosity,
        store=True,
        include=["reasoning.encrypted_content"],
        tools=[],
    )

    context = """\
Alice has a cat named Snowball.
She also has a dog named Rex.
"""
    query = "What is the cat's name?"

    rlm = RLM(root_llm=root_client, sub_llm=sub_client)
    answer = rlm.answer(query, context)
    print(answer)


if __name__ == "__main__":
    main()
