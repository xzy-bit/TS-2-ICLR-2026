import os
import sys
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast, TypeVar, Sequence, Iterable

from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from tqdm.auto import tqdm
from transformers import HfArgumentParser, AutoTokenizer

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, ".."))
)

from evaluation.utils.magicoder import (
    GenerationConfig,
    get_model_context,
    TokenizationContext,
)

def get_tokenizer(tokenizer_name_or_path, **kwargs):
    return AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, **kwargs)

# MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

# @@ Instruction
# {instruction}

# @@ Response
# {response}"""

MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}
"""

# MAGICODER_PROMPT = "{instruction}"


class Text2CodeProblem(TypedDict):
    id: str
    instruction: str
    response_prefix: str


MBPP_INSTRUCTION = """{nl_description} Your code should satisfy the following assertion:
```python
{assertions}
```
Enclose your solution in ```python and ```"""


_T = TypeVar("_T")


def chunked(seq: Sequence[_T], n: int) -> Iterable[Sequence[_T]]:
    """Yield successive n-sized chunks from seq."""
    return (seq[i : i + n] for i in range(0, len(seq), n))


def get_mbpp_raw_problems() -> list[dict]:
    problems = get_mbpp_plus()
    return list(problems.values())


def get_humaneval_raw_problems() -> list[dict]:
    problems = get_human_eval_plus()
    return list(problems.values())


def map_mbpp_problem(p: dict) -> Text2CodeProblem:
    id = p["task_id"]
    prompt = p["prompt"]
    start_index = prompt.index('"""')
    end_index = prompt.rindex('"""')
    prompt = prompt[start_index + 3 : end_index]
    assert_index = prompt.index("assert")
    instruction = prompt[:assert_index].strip()
    if not instruction.endswith("."):
        instruction += "."
    assertion = prompt[assert_index:].strip()
    instruction = f"""{instruction} Your code should satisfy the following assertion:
```python
{assertion}
```"""
    response_prefix = f"""```python"""
    return Text2CodeProblem(
        id=str(id), instruction=instruction, response_prefix=response_prefix
    )


def map_humaneval_problem(p: dict) -> Text2CodeProblem:
    id = p["task_id"]
    prompt = p["prompt"]
    prompt = prompt.strip()
    # try:
    #     docstring_index = prompt.index('"""')
    # except ValueError:
    #     docstring_index = prompt.index("'''")
    # signature = prompt[:docstring_index].strip()
    # Instruction
    instruction = f"""Write a solution to the following problem:
```python
{prompt}
```"""
    response_prefix = f"""```python
    {prompt}"""
    # response_prefix = f"""```python"""
    return Text2CodeProblem(
        id=id, instruction=instruction, response_prefix=response_prefix
    )


def map_humaneval_problem_simple(p: dict) -> Text2CodeProblem:
    id = p["task_id"]
    prompt = p["prompt"]
    prompt = prompt.strip()
    # try:
    #     docstring_index = prompt.index('"""')
    # except ValueError:
    #     docstring_index = prompt.index("'''")
    # signature = prompt[:docstring_index].strip()
    # Instruction
    instruction = f"""Write a solution to the following problem:
```python
{prompt}
```"""
    # response_prefix = f"""```python
    # {prompt}"""
    response_prefix = f"""```python"""
    return Text2CodeProblem(
        id=id, instruction=instruction, response_prefix=response_prefix
    )


@dataclass(frozen=True)
class Args:
    model_key: str
    dataset: Literal["humaneval", "mbpp"]
    save_path: str

    n_batches: int
    n_problems_per_batch: int
    n_samples_per_problem: int
    # prompted: bool

    model_name_or_path: str | None = None
    tokenizer_name_or_path: str | None = None

    use_vllm: bool = True
    template: str = "normal"


def main():
    parser = HfArgumentParser((Args, GenerationConfig))
    args, generation_config = cast(
        tuple[Args, GenerationConfig],
        parser.parse_args_into_dataclasses(),
    )
    raw_problem_fn, map_problem_fn = (
        (
            get_humaneval_raw_problems,
            (
                map_humaneval_problem
                if args.template == "normal"
                else map_humaneval_problem_simple
            ),
        )
        if args.dataset == "humaneval"
        else (get_mbpp_raw_problems, map_mbpp_problem)
    )
    raw_problems = raw_problem_fn()
    problems = list(map(map_problem_fn, raw_problems))

    tokenizer = get_tokenizer(args.tokenizer_name_or_path)
    tokenization_context = TokenizationContext.from_tokenizer(tokenizer)
    state = get_model_context(
        args.model_key,
        args.model_name_or_path,
        tokenization_context,
        use_vllm=args.use_vllm,
    )

    problems_chunked = list(chunked(list(problems), args.n_problems_per_batch))
    iter = itertools.product(problems_chunked, range(args.n_batches))
    n_total = len(problems_chunked) * args.n_batches

    Path(args.save_path).write_text("")
    for problems, batch_idx in tqdm(iter, total=n_total):
        task_ids = [problem["id"] for problem in problems]
        prompts = []
        for problem in problems:
            # MAGICODER_PROMPT.format(
            #     instruction=problem["instruction"], response=problem["response_prefix"]
            # )
            prompt_token = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": MAGICODER_PROMPT.format(
                            instruction=problem["instruction"]
                        ),
                    }
                ],
                add_generation_prompt=True,
            )
            prompt_text = tokenizer.decode(prompt_token) + problem["response_prefix"]
            prompts.append(prompt_text)
        print("=======PROMPT=======")
        print(prompts[-1])
        all_prompts = prompts * args.n_samples_per_problem
        all_task_ids = task_ids * args.n_samples_per_problem
        response = state.complete(generation_config, all_prompts)
        completions = response.decoded_outputs
        assert len(problems) <= args.n_problems_per_batch
        assert len(completions) == len(problems) * args.n_samples_per_problem
        print("======COMPLETION======")
        print(completions[-1])
        print()
        samples = [
            dict(
                task_id=task_id,
                completion=completion[
                    : (
                        index
                        if (index := completion.find("```")) != -1
                        else len(completion)
                    )
                ],
            )
            for task_id, completion in zip(all_task_ids, completions)
        ]
        write_jsonl(args.save_path, samples, append=True)


if __name__ == "__main__":
    main()
