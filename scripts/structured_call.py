"""Try to get structured output out of the model, the naive way.

    python scripts/structured_call.py

The user message deliberately asks the model to "think first" - a normal-sounding
instruction that makes it emit prose before the JSON, which breaks json.loads.
"""
from pydantic import BaseModel

from sec10k.llm import chat_structured


class ItemSummary(BaseModel):
    item: str
    title: str
    one_line_purpose: str


def main() -> None:
    obj, result = chat_structured(
        [
            {
                "role": "user",
                "content": "Think through it briefly, then describe Item 7 of a Form 10-K.",
            }
        ],
        ItemSummary,
        tag="structured-demo",
    )
    print("parsed :", obj)
    print("tokens :", result.record.total_tokens)


if __name__ == "__main__":
    main()
