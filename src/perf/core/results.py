
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class RequestMetrics:
    user_id: int
    user_type: str
    tier: Optional[str] = None
    num_run: int = 0
    timestamp: float = field(default_factory=time.time)

    ttft: float = 0.0
    tpot: float = 0.0
    latency: float = 0.0

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    itl: List[float] = field(default_factory=list)

    prompt: str = ""
    answer: Optional[str] = None

    status: int = 0
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.error is None and self.status == 200

    @property
    def ttft_ms(self) -> float:
        return self.ttft * 1000

    @property
    def tpot_ms(self) -> float:
        return self.tpot * 1000

    @property
    def itl_ms(self) -> List[float]:
        return [x * 1000 for x in self.itl]

    def cost(
        self,
        input_cost_per_million: float = 0.15,
        output_cost_per_million: float = 0.60,
    ) -> float:
        input_cost = (self.prompt_tokens / 1000000) * input_cost_per_million
        output_cost = (self.completion_tokens / 1000000) * output_cost_per_million
        return input_cost + output_cost


@dataclass
class TokenEvent:
    user_id: int
    user_type: str
    num_run: int
    token_idx: int
    timestamp: float
    wall_time: str
    itl_ms: float
    token_text: str
    kv_usage: str = ""


__all__ = [
    "RequestMetrics",
    "TokenEvent",
]
