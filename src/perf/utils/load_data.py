
import os
from datasets import load_dataset
from .template import (
    Prompts,
    ConversationTemplate,
    AlpacaTemplate,
    PlainTemplate,
    ChatMLTemplate,
    LoopllmTemplate,
    EngorgioTemplate,
    LongbenchTemplate,
    DATemplate,
    ExtendAttackTemplate,
    OverthinkTemplate,
    FsAttackTemplate,
)
import random
from typing import Optional

def get_template(template_name: str, dataset_name: str) -> ConversationTemplate:
    if template_name == "plain":
        return PlainTemplate()
    elif template_name == "alpaca":
        return AlpacaTemplate()
    elif template_name == "chatml":
        return ChatMLTemplate()
    elif template_name == "loopllm":
        return LoopllmTemplate()
    elif template_name == "engorgio":
        return EngorgioTemplate()
    # elif template_name == "longbench":
    #     return LongbenchTemplate()
    elif template_name == "fs_attack":
        return FsAttackTemplate()
    elif "extend_attack" in template_name:
        ratio = float(template_name.split("extend_attack")[1])
        return ExtendAttackTemplate(
            ratio=ratio, dataset_name=dataset_name,model_name="qwen")
    else:
        raise ValueError(f"Unsupported template name: {template_name}")


def load_prompts(
    dataset_path: str,
    template_name: str,
    max_prompts: Optional[int] = None,
) -> list[Prompts]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    if dataset_path.endswith(".json"):
        datasets = load_dataset("json", data_files=dataset_path)["train"]
    elif dataset_path.endswith(".txt"):
        datasets = load_dataset("text", data_files=dataset_path)["train"]
    elif dataset_path.endswith(".csv"):
        datasets = load_dataset("csv", data_files=dataset_path)["train"]
    elif dataset_path.endswith(".jsonl"):
        datasets = load_dataset("json", data_files=dataset_path)["train"]
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")

    template = get_template(template_name, dataset_path)

    if max_prompts is not None:
        idx = random.sample(range(len(datasets)), min(max_prompts, len(datasets)))
        datasets = datasets.select(idx)
    prompts = [template.format_prompt(item) for item in datasets]

    return prompts
