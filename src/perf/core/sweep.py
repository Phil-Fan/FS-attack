
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class PayloadConfig:
    name: str
    file: str
    template: str = "plain"
    attack_method: str = ""
    attack_strength: str = ""


@dataclass
class SweepConfig:

    output_dir: str = "output/sweeps"

    ratios: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])

    repeats: int = 1

    payloads: List[PayloadConfig] = field(default_factory=list)

    base_overrides: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        for payload in self.payloads:
            for ratio in self.ratios:
                for rep in range(1, self.repeats + 1):
                    yield SweepIteration(
                        payload=payload,
                        ratio=ratio,
                        repeat=rep,
                        sweep_config=self,
                    )


@dataclass
class SweepIteration:

    payload: PayloadConfig
    ratio: float
    repeat: int
    sweep_config: SweepConfig

    @property
    def run_id(self) -> str:
        ratio_tag = str(self.ratio).replace(".", "p")
        file_tag = Path(self.payload.file).stem
        return f"payload_{self.payload.name}__file_{file_tag}__ratio_{ratio_tag}_rep_{self.repeat}"


def get_benchmark_sweep() -> SweepConfig:
    base_path = Path(__file__).parent.parent.parent / "datasets"

    return SweepConfig(
        output_dir="output/sweeps/benchmark",
        ratios=[0, 0.25, 0.5, 0.75],
        repeats=1,
        payloads=[
            PayloadConfig(
                name="loopllm",
                file=str(base_path / "loopllm" / "loopllm_8B.json"),
                template="loopllm",
            ),
            PayloadConfig(
                name="engorgio",
                file=str(base_path / "engorgio" / "Qwen3_8B.jsonl"),
                template="engorgio",
            ),
        ],
    )


def get_quick_test_sweep() -> SweepConfig:
    base_path = Path(__file__).parent.parent.parent / "datasets"

    return SweepConfig(
        output_dir="output/sweeps/quick_test",
        ratios=[0, 0.5],
        repeats=1,
        payloads=[
            PayloadConfig(
                name="test",
                file=str(base_path / "benign" / "alpaca.json"),
                template="alpaca",
            ),
        ],
        base_overrides={
            "users": {"total": 5},
            "experiment": {"duration": 10},
        },
    )


def get_custom_sweep(
    ratios: List[float],
    payloads: List[Dict[str, Any]],
    repeats: int = 1,
    **overrides
) -> SweepConfig:
    payload_objs = [
        PayloadConfig(
            name=p["name"],
            file=p["file"],
            template=p.get("template", "plain"),
            attack_method=p.get("attack_method", p["name"]),
            attack_strength=p.get("attack_strength", ""),
        )
        for p in payloads
    ]

    return SweepConfig(
        ratios=ratios,
        repeats=repeats,
        payloads=payload_objs,
        base_overrides=overrides,
    )


__all__ = [
    "PayloadConfig",
    "SweepConfig",
    "SweepIteration",
    "get_benchmark_sweep",
    "get_quick_test_sweep",
    "get_custom_sweep",
]
