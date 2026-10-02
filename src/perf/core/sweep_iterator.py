
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterator, List


@dataclass
class SweepCombo:
    duration: int
    normal_dataset: str
    special_type: str
    special_dataset: str  # Legacy format: a single dataset name
    special_ratio: float  # Special user ratio (used by legacy mode)
    prober_count: int
    repeat: int
    combo_id: str
    sweep_name: str
    metadata: Dict[str, Any]
    normal_count: int = None
    malicious_count: int = None
    special_datasets_tiered: Dict[str, List[str]] = None  # New format: {"high": [...], "mid": [...], "low": [...]}

    def to_config(self, base_config: Dict) -> Dict:
        import copy
        cfg = copy.deepcopy(base_config)

        cfg.setdefault("experiment", {})["duration"] = self.duration

        cfg.setdefault("users", {})["ratios"] = {
            self.special_type: self.special_ratio,
            "prober": self.prober_count,
        }

        if self.normal_count is not None and self.malicious_count is not None:
            total = self.normal_count + self.malicious_count + self.prober_count
            cfg.setdefault("users", {})["total"] = total
            cfg["users"]["ratios"] = {
                self.special_type: self.malicious_count,
                "prober": self.prober_count,
            }

        cfg.setdefault("datasets", {})["normal"] = self.normal_dataset

        if self.special_datasets_tiered:
            cfg["datasets"][self.special_type] = self.special_datasets_tiered
        else:
            cfg["datasets"][self.special_type] = self.special_dataset

        cfg.setdefault("sweep", {})
        cfg["sweep"].update(self.metadata)
        cfg["sweep"]["combo_id"] = self.combo_id
        cfg["sweep"]["repeat"] = self.repeat

        return cfg


@dataclass
class SweepConfig:
    name: str
    enabled: bool
    output_dir: str
    durations: List[int]
    datasets: Dict[str, List]
    special_ratios: Dict[str, List[float]]  # Legacy mode: special user ratio
    prober_counts: List[int]
    repeats: int
    base_overrides: Dict
    malicious_counts: List[int] = None
    normal_counts: List[int] = None

    def total_count(self) -> int:
        use_new_mode = (
            self.malicious_counts is not None and 
            self.normal_counts is not None and
            len(self.malicious_counts) > 0 and
            len(self.normal_counts) > 0
        )
        
        if use_new_mode:
            normal_ds_count = len(self.datasets.get("normal", ["benign.alpaca"]))
            special_ds_count = len(self.datasets.get("fs_attack", [])) + len(self.datasets.get("baseline", []))
            
            total = (
                len(self.durations) *
                normal_ds_count *
                special_ds_count *
                len(self.malicious_counts) *
                len(self.normal_counts) *
                len(self.prober_counts) *
                self.repeats
            )
            return total
        
        # === Legacy mode: use special_ratios (ratio mode) ===
        experiment_types_count = 0
        for baseline_ds in self.datasets.get("baseline", []):
            for baseline_ratio in self.special_ratios.get("baseline", [0]):
                experiment_types_count += 1

        fs_attack_cfg = self.datasets.get("fs_attack", [])
        if isinstance(fs_attack_cfg, dict) and any(tier in fs_attack_cfg for tier in ("high", "mid", "low")):
            for fs_attack_ratio in self.special_ratios.get("fs_attack", [0]):
                experiment_types_count += 1
        else:
            for fs_attack_ds in fs_attack_cfg:
                for fs_attack_ratio in self.special_ratios.get("fs_attack", [0]):
                    experiment_types_count += 1
        
        normal_ds_count = len(self.datasets.get("normal", ["benign.alpaca"]))
        
        total = (
            len(self.durations) *
            normal_ds_count *
            experiment_types_count *
            len(self.prober_counts) *
            self.repeats
        )
        return total

        total = (
            len(self.durations) *
            normal_ds_count *
            experiment_types_count *
            len(self.prober_counts) *
            self.repeats
        )
        return total

    def __iter__(self) -> Iterator[SweepCombo]:
        
        use_new_mode = (
            self.malicious_counts is not None and 
            self.normal_counts is not None and
            len(self.malicious_counts) > 0 and
            len(self.normal_counts) > 0
        )
        
        if use_new_mode:
            for duration in self.durations:
                for normal_ds in self.datasets.get("normal", ["benign.alpaca"]):
                    for special_ds in self.datasets.get("fs_attack", []) + \
                                      self.datasets.get("baseline", []):
                        special_type = "fs_attack" if special_ds in self.datasets.get("fs_attack", []) or \
                                                  (isinstance(special_ds, list) and "fs_attack.tiered" in special_ds) else "baseline"
                        
                        for malicious_count in self.malicious_counts:
                            for normal_count in self.normal_counts:
                                for prober_count in self.prober_counts:
                                    for repeat in range(1, self.repeats + 1):
                                        total = normal_count + malicious_count + prober_count
                                        special_ratio = malicious_count / total if total > 0 else 0
                                        
                                        combo_id = f"{duration}s_mal{malicious_count}_norm{normal_count}"
                                        if repeat > 1:
                                            combo_id += f"_r{repeat}"
                                        
                                        yield SweepCombo(
                                            duration=duration,
                                            normal_dataset=normal_ds,
                                            special_type=special_type,
                                            special_dataset=special_ds,
                                            special_ratio=special_ratio,
                                            prober_count=prober_count,
                                            repeat=repeat,
                                            combo_id=combo_id,
                                            sweep_name=self.name,
                                            metadata={
                                                "sweep_name": self.name,
                                                "duration": duration,
                                                "normal_dataset": normal_ds,
                                                "special_type": special_type,
                                                "special_dataset": special_ds,
                                                "special_ratio": special_ratio,
                                                "normal_count": normal_count,
                                                "malicious_count": malicious_count,
                                                "mode": "fixed_count",
                                            },
                                            normal_count=normal_count,
                                            malicious_count=malicious_count,
                                        )
            return
        
        # === Legacy mode: use special_ratios (ratio mode) ===
        experiment_types = []
        for baseline_ds in self.datasets.get("baseline", []):
            for baseline_ratio in self.special_ratios.get("baseline", [0]):
                experiment_types.append(("baseline", baseline_ds, baseline_ratio, None))

        fs_attack_cfg = self.datasets.get("fs_attack", [])
        if isinstance(fs_attack_cfg, dict) and any(tier in fs_attack_cfg for tier in ("high", "mid", "low")):
            tiered_datasets = {
                "high": fs_attack_cfg.get("high", []),
                "mid": fs_attack_cfg.get("mid", []),
                "low": fs_attack_cfg.get("low", []),
            }
            for fs_attack_ratio in self.special_ratios.get("fs_attack", [0]):
                experiment_types.append(("fs_attack", "fs_attack_tiered", fs_attack_ratio, tiered_datasets))
        else:
            # Legacy format: list
            for fs_attack_ds in fs_attack_cfg:
                for fs_attack_ratio in self.special_ratios.get("fs_attack", [0]):
                    experiment_types.append(("fs_attack", fs_attack_ds, fs_attack_ratio, None))

        for duration in self.durations:
            for normal_ds in self.datasets.get("normal", ["benign.alpaca"]):
                for special_type, special_ds, special_ratio, tiered_datasets in experiment_types:
                    for prober_count in self.prober_counts:
                        for repeat in range(1, self.repeats + 1):
                            # Convert loopllm.qwen_8b to loopllm_qwen_8b
                            if tiered_datasets:
                                ds_name = "fs_attack_tiered"
                            elif isinstance(special_ds, list):
                                ds_name = f"{special_type}_all"
                            elif "." in special_ds:
                                parts = special_ds.split(".")
                                if len(parts) >= 2:
                                    ds_name = "_".join(parts[-2:])
                                else:
                                    ds_name = parts[-1]
                            else:
                                ds_name = special_ds
                            combo_id = f"{duration}s_{ds_name}_{int(special_ratio * 100)}p"
                            if repeat > 1:
                                combo_id += f"_r{repeat}"

                            yield SweepCombo(
                                duration=duration,
                                normal_dataset=normal_ds,
                                special_type=special_type,
                                special_dataset=special_ds,
                                special_ratio=special_ratio,
                                prober_count=prober_count,
                                repeat=repeat,
                                combo_id=combo_id,
                                sweep_name=self.name,
                                metadata={
                                    "sweep_name": self.name,
                                    "duration": duration,
                                    "normal_dataset": normal_ds,
                                    "special_type": special_type,
                                    "special_dataset": special_ds,
                                    "special_ratio": special_ratio,
                                    "mode": "ratio",
                                    "tiered_datasets": tiered_datasets,
                                },
                                special_datasets_tiered=tiered_datasets,
                            )


def load_sweep(sweep_name: str, loader) -> SweepConfig:
    sweeps_cfg = loader._load_sweeps()
    sweeps = sweeps_cfg.get("sweeps", {})

    if sweep_name not in sweeps:
        raise ValueError(f"Sweep not found: {sweep_name}")

    sweep_cfg = sweeps[sweep_name]

    malicious_counts = sweep_cfg.get("malicious_counts")
    normal_counts = sweep_cfg.get("normal_counts")

    return SweepConfig(
        name=sweep_name,
        enabled=sweep_cfg.get("enabled", True),
        output_dir=sweep_cfg.get("output_dir", "output/sweeps"),
        durations=sweep_cfg.get("durations", [60]),
        datasets=sweep_cfg.get("datasets", {}),
        special_ratios=sweep_cfg.get("special_ratios", {}),
        prober_counts=sweep_cfg.get("prober_counts", [0]),
        repeats=sweep_cfg.get("repeats", 1),
        base_overrides=sweep_cfg.get("base_overrides", {}),
        malicious_counts=malicious_counts,
        normal_counts=normal_counts,
    )


def list_sweeps(loader) -> Dict[str, Dict]:
    sweeps_cfg = loader._load_sweeps()
    result = {}
    for name, cfg in sweeps_cfg.get("sweeps", {}).items():
        result[name] = {
            "enabled": cfg.get("enabled", False),
            "description": cfg.get("description", ""),
        }
    return result


__all__ = [
    "SweepCombo",
    "SweepConfig",
    "load_sweep",
    "list_sweeps",
]
