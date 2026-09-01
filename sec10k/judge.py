"""LLM judge: does a predicted answer match the reference answer?

The eval metric depends on this call, so it gets calibrated against hand labels
(scripts/calibrate_judge.py) before its verdicts are trusted.
"""
from __future__ import annotations

from pydantic import BaseModel

from sec10k.goldens import REFUSAL
from sec10k.llm import chat_structured

JUDGE_SYSTEM = (
    "You grade whether a PREDICTED answer to a question about an SEC 10-K filing "
    "matches the REFERENCE answer. Judge the key fact or figure, not the wording.\n"
    "- Numbers equal but formatted differently match "
    '("$383.3 billion" == "$383,285 million").\n'
    f"- If the reference is {REFUSAL}, the prediction is correct only if it also "
    "declines / says the information is not in the filing.\n"
    "- Extra correct context is fine; a wrong or missing key figure is not.\n"
    "Return: correct (bool), score (0.0-1.0), reasoning (one sentence)."
)


class Verdict(BaseModel):
    correct: bool
    score: float
    reasoning: str


def judge(question: str, gold: str, predicted: str, *, tag: str = "judge") -> Verdict:
    """Ask the model to grade `predicted` against `gold` for `question`.

    Steps:
      1. messages = [
             {"role": "system", "content": JUDGE_SYSTEM},
             {"role": "user", "content":
                 f"Question: {question}\n\n"
                 f"Reference answer: {gold}\n\n"
                 f"Predicted answer: {predicted}"},
         ]
      2. verdict, _ = chat_structured(messages, Verdict, tag=tag)
      3. return verdict
    """
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Reference answer: {gold}\n\n"
                f"Predicted answer: {predicted}"
            ),
        },
    ]
    verdict, _ = chat_structured(messages, Verdict, tag=tag)
    return verdict
