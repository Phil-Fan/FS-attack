#!/usr/bin/env python3

import argparse
import sys
import logging
from pathlib import Path

from config import (
    ProbeConfig,
    get_default_output_dir,
    setup_logging,
    check_dependencies
)
from tasks import (
    TaskConfig,
    TrainTask,
    InferTask,
    BenchmarkTask
)

logger = logging.getLogger(__name__)


def create_train_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("train", help="Train models")

    parser.add_argument("data_dir", help="Data directory")
    parser.add_argument("--model", "-m", default="all",
                        choices=["minirocket", "inceptiontime", "lightgbm", "all"],
                        help="Model type (default: all)")
    parser.add_argument("--epochs", "-e", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", "-b", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory")
    parser.add_argument("--device", default="auto",
                        help="Device (cpu/cuda/auto)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    return parser


def create_infer_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("infer", help="Model inference")

    parser.add_argument("data_dir", help="Data directory")
    parser.add_argument("--model-dir", "-m", type=Path, required=True,
                        help="Model directory")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory")
    parser.add_argument("--save-predictions", action="store_true", default=True,
                        help="Save prediction results")

    return parser


def create_benchmark_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("benchmark", help="Benchmark")

    parser.add_argument("data_dir", help="Data directory")
    parser.add_argument("--models", "-m", default="all",
                        help="Comma-separated list of models (default: all)")
    parser.add_argument("--epochs", "-e", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", "-b", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory")

    return parser


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe: Time Series Classification Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  python cli.py train /path/to/data --model minirocket --epochs 50

  # Inference
  python cli.py infer /path/to/data --model-dir /path/to/model

  # Benchmark
  python cli.py benchmark /path/to/data --models minirocket,inceptiontime

  # Generalization test (standalone script)
  python generalization.py --help

For more information, visit: https://github.com/your-org/latency-attack
        """
    )

    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--version", action="version",
                        version="%(prog)s 1.0.0")

    subparsers = parser.add_subparsers(dest="task", required=True,
                                        help="Task type")

    create_train_parser(subparsers)
    create_infer_parser(subparsers)
    create_benchmark_parser(subparsers)

    return parser.parse_args()


def run_train(args) -> int:
    logger.info(f"Starting training: {args.data_dir}")

    data_dir = Path(args.data_dir)
    output_dir = args.output or get_default_output_dir("train", data_dir)

    config = TaskConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        device=args.device,
        seed=args.seed
    )

    task = TrainTask(
        config=config,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    result = task.run()

    if result.success:
        logger.info("Training complete!")
        task._log_metrics()
        return 0
    else:
        logger.error(f"Training failed: {result.error_message}")
        return 1


def run_infer(args) -> int:
    logger.info(f"Starting inference: {args.data_dir}")

    data_dir = Path(args.data_dir)
    output_dir = args.output or get_default_output_dir("infer", data_dir)

    config = TaskConfig(
        data_dir=data_dir,
        output_dir=output_dir
    )

    task = InferTask(
        config=config,
        model_dir=args.model_dir,
        output_predictions=args.save_predictions
    )

    result = task.run()

    if result.success:
        logger.info("Inference complete!")
        task._log_metrics()
        return 0
    else:
        logger.error(f"Inference failed: {result.error_message}")
        return 1


def run_benchmark(args) -> int:
    logger.info(f"Starting benchmark: {args.data_dir}")

    if args.models == "all":
        model_types = ["minirocket", "inceptiontime", "lightgbm"]
    else:
        model_types = [m.strip() for m in args.models.split(",")]

    data_dir = Path(args.data_dir)
    output_dir = args.output or get_default_output_dir("benchmark", data_dir)

    config = TaskConfig(
        data_dir=data_dir,
        output_dir=output_dir
    )

    task = BenchmarkTask(
        config=config,
        model_types=model_types,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    result = task.run()

    if result.success:
        logger.info("Benchmark complete!")
        task._log_metrics()
        return 0
    else:
        logger.error(f"Benchmark failed: {result.error_message}")
        return 1


def main():
    args = parse_args()

    setup_logging(verbose=args.verbose)

    check_dependencies()

    task_handlers = {
        "train": run_train,
        "infer": run_infer,
        "benchmark": run_benchmark,
    }

    handler = task_handlers.get(args.task)
    if handler:
        sys.exit(handler(args))
    else:
        logger.error(f"Unknown task: {args.task}")
        sys.exit(1)


if __name__ == "__main__":
    main()
