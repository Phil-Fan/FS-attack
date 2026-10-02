
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import json


@dataclass
class ProbeConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")

    model_type: str = "all"
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3

    num_classes: int = 0

    device: str = "auto"
    seed: int = 42
    verbose: bool = True

    run_output_dir: str = field(default_factory=lambda: os.environ.get("RUN_OUTPUT_DIR", "runs"))

    def to_dict(self) -> Dict:
        return {
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "model_type": self.model_type,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_classes": self.num_classes,
            "device": self.device,
            "seed": self.seed,
            "verbose": self.verbose,
            "run_output_dir": self.run_output_dir
        }

    def save(self, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "ProbeConfig":
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(
            data_dir=Path(data.get("data_dir", "data")),
            output_dir=Path(data.get("output_dir", "outputs")),
            model_type=data.get("model_type", "all"),
            epochs=data.get("epochs", 50),
            batch_size=data.get("batch_size", 64),
            learning_rate=data.get("learning_rate", 1e-3),
            num_classes=data.get("num_classes", 0),
            device=data.get("device", "auto"),
            seed=data.get("seed", 42),
            verbose=data.get("verbose", True),
            run_output_dir=data.get("run_output_dir", "runs")
        )


def get_default_output_dir(task: str, data_dir: Path) -> Path:
    timestamp = get_timestamp()
    output_dir = Path("outputs") / task / data_dir.name / timestamp
    return output_dir


def get_timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(verbose: bool = True):
    import logging

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def check_dependencies():
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import tsai
    except ImportError:
        pass

    try:
        import lightgbm
    except ImportError:
        pass

    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn")

    try:
        import pandas
    except ImportError:
        missing.append("pandas")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    if missing:
        print(f"Warning: the following dependencies are not installed: {', '.join(missing)}")
        print("Please run: pip install " + " ".join(missing))

    return len(missing) == 0
