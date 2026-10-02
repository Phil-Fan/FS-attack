
from .base_task import BaseTask, TaskConfig, TaskResult
from .train_task import TrainTask
from .infer_task import InferTask
from .benchmark_task import BenchmarkTask

__all__ = [
    'BaseTask',
    'TaskConfig',
    'TaskResult',
    'TrainTask',
    'InferTask',
    'BenchmarkTask'
]
