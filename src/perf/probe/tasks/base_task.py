
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    data_dir: Path
    output_dir: Optional[Path] = None
    device: str = "auto"
    seed: int = 42


@dataclass
class TaskResult:
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Path] = field(default_factory=dict)
    duration_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class BaseTask(ABC):

    def __init__(self, config: TaskConfig):
        self.config = config
        self.start_time: float = 0.0
        self.result = TaskResult()

    def run(self) -> TaskResult:
        self.start_time = time.perf_counter()

        try:
            self._before_run()
            self._execute()
            self._after_run()
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            self.result.success = False
            self.result.error_message = str(e)
            raise
        finally:
            self.result.duration_seconds = time.perf_counter() - self.start_time
            logger.info(f"Task complete, elapsed time: {self.result.duration_seconds:.2f} seconds")

        return self.result

    def _before_run(self):
        if self.config.output_dir:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {self.config.output_dir}")

    @abstractmethod
    def _execute(self):
        pass

    def _after_run(self):
        pass

    def _save_metrics(self, filename: str = "metrics.json"):
        if self.config.output_dir and self.result.metrics:
            path = self.config.output_dir / filename
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.result.metrics, f, indent=2, ensure_ascii=False)
            logger.info(f"Metrics saved: {path}")

    def _log_metrics(self, prefix: str = ""):
        for key, value in self.result.metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{prefix}{key}: {value:.4f}" if isinstance(value, float) else f"{prefix}{key}: {value}")


def create_task(task_type: str, **kwargs) -> BaseTask:
    from .train_task import TrainTask
    from .infer_task import InferTask
    from .benchmark_task import BenchmarkTask

    task_classes = {
        "train": TrainTask,
        "infer": InferTask,
        "benchmark": BenchmarkTask,
    }

    if task_type not in task_classes:
        raise ValueError(f"Unknown task type: {task_type}")

    return task_classes[task_type](**kwargs)
