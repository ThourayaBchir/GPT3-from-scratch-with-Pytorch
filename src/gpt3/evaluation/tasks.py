"""Local JSONL multiple-choice tasks scored by continuation likelihood."""

import math
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tokenizers import Tokenizer

from gpt3.config import TaskConfig
from gpt3.model import GPT3


class TaskExample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str = Field(min_length=1)
    choices: list[str] = Field(min_length=2)
    answer: int = Field(ge=0)

    @model_validator(mode="after")
    def check_answer(self) -> "TaskExample":
        if self.answer >= len(self.choices) or any(not choice for choice in self.choices):
            raise ValueError("Choices must be nonempty and answer must index a choice")
        return self


def read_examples(path: str) -> list[TaskExample]:
    examples = [
        TaskExample.model_validate_json(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if not examples:
        raise ValueError(f"Task file {path} has no examples")
    return examples


def make_prompt(task: TaskConfig, example: TaskExample, demos: list[TaskExample]) -> str:
    prefix = task.instruction + "\n\n" if task.instruction else ""
    for demo in demos[: task.shots]:
        prefix += demo.prompt + demo.choices[demo.answer] + "\n\n"
    return prefix + example.prompt


def _tokenize_choices(
    tokenizer: Tokenizer, prompt: str, choices: list[str]
) -> tuple[list[int], list[list[int]]]:
    prompt_ids = tokenizer.encode(prompt.rstrip(), add_special_tokens=False).ids
    choice_ids = [
        tokenizer.encode(prompt + choice, add_special_tokens=False).ids for choice in choices
    ]

    # Score any tokens that the tokenizer re-forms at the boundary. Every
    # choice is conditioned on the same stable part of the prompt.
    prefix_length = 0
    for prompt_id, *candidate_ids in zip(prompt_ids, *choice_ids, strict=False):
        if any(candidate_id != prompt_id for candidate_id in candidate_ids):
            break
        prefix_length += 1
    if prefix_length == 0:
        raise ValueError("Task prompt must provide a shared token before the choices")
    return prompt_ids[:prefix_length], [ids[prefix_length:] for ids in choice_ids]


@torch.no_grad()
def continuation_score(
    model: GPT3, prompt_ids: list[int], completion_ids: list[int], *, length_normalize: bool = False
) -> float:
    """Score the completion tokens conditioned on the supplied prompt tokens."""
    if not prompt_ids or not completion_ids:
        raise ValueError("Prompts and choices must encode to at least one token")
    tokens = prompt_ids + completion_ids
    if len(tokens) - 1 > model.config.context_length:
        raise ValueError(
            "Task prompt plus choice exceeds model context; reduce shots or text length"
        )
    device = next(model.parameters()).device
    inputs = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    labels = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
    labels[:, : len(prompt_ids) - 1] = -100
    was_training = model.training
    model.eval()
    try:
        result = model(inputs, labels=labels, return_logits=False)
        assert result.loss is not None
        score = -float(result.loss) * (1 if length_normalize else len(completion_ids))
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite task score")
        return score
    finally:
        model.train(was_training)


def evaluate_task(model: GPT3, tokenizer: Tokenizer, task: TaskConfig) -> dict[str, Any]:
    examples = read_examples(task.evaluation_file)
    demos: list[TaskExample] = []
    if task.shots:
        assert task.demonstrations_file is not None
        if Path(task.evaluation_file).resolve() == Path(task.demonstrations_file).resolve():
            raise ValueError("Demonstrations must come from a separate file")
        demos = read_examples(task.demonstrations_file)
        if len(demos) < task.shots:
            raise ValueError("Not enough demonstrations for configured shots")
        if {demo.prompt for demo in demos[: task.shots]} & {ex.prompt for ex in examples}:
            raise ValueError("Demonstration and evaluation prompts overlap")
    predictions = []
    for example in examples:
        prompt = make_prompt(task, example, demos)
        prompt_ids, completions = _tokenize_choices(tokenizer, prompt, example.choices)
        scores = [
            continuation_score(
                model,
                prompt_ids,
                completion_ids,
                length_normalize=task.length_normalize,
            )
            for completion_ids in completions
        ]
        predicted = max(range(len(scores)), key=scores.__getitem__)
        predictions.append(
            {
                "prompt": example.prompt,
                "predicted": predicted,
                "answer": example.answer,
                "scores": scores,
            }
        )
    return {
        "name": task.name,
        "shots": task.shots,
        "examples": len(examples),
        "accuracy": sum(p["predicted"] == p["answer"] for p in predictions) / len(examples),
        "predictions": predictions,
    }
