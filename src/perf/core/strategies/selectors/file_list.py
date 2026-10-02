
from __future__ import annotations

from typing import Dict, List

from .base import PromptSelector


class FileListSelector(PromptSelector):

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        prompts = config.get("prompts", [])
        if isinstance(prompts, str):
            try:
                with open(prompts, "r", encoding="utf-8") as f:
                    self.prompts = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                raise ValueError(f"Prompt file not found: {prompts}")
        else:
            self.prompts = list(prompts) if prompts else []

        if not self.prompts:
            raise ValueError("FileListSelector requires at least one prompt")

        self.shuffle = config.get("shuffle", False)
        self.repeat = config.get("repeat", True)
        self._index = 0

        if self.shuffle:
            import random

            random.shuffle(self.prompts)

    def get_pool(self, prompt_tiers: Dict[str, List[str]]) -> List[str]:
        return self.prompts

    def select_one(self, prompt_tiers: Dict[str, List[str]]) -> str:
        if not self.prompts:
            raise ValueError("No prompts available in FileListSelector")

        prompt = self.prompts[self._index]

        self._index += 1
        if self._index >= len(self.prompts):
            if self.repeat:
                self._index = 0
                if self.shuffle:
                    import random

                    random.shuffle(self.prompts)
            else:
                self._index = len(self.prompts) - 1

        return prompt
