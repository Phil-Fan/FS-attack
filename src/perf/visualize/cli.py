#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

try:
    from utils.logging_utils import setup_logging
except ImportError:
    import os as _os

    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from utils.logging_utils import setup_logging

try:
    from .paths import (
        collect_results_csvs,
        collect_run_dirs,
        filter_by_sweep_timestamp,
        pick_latest_sweep_timestamp,
        read_run_dir_from_env_or_file,
        find_latest_scheduler_log,
    )
    from .summary import summarize_runs
    from . import output_side
    from . import slo_side
    from . import sys_side
except ImportError:
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paths import (
        collect_results_csvs,
        collect_run_dirs,
        filter_by_sweep_timestamp,
        pick_latest_sweep_timestamp,
        read_run_dir_from_env_or_file,
        find_latest_scheduler_log,
    )
    from summary import summarize_runs
    import output_side
    import slo_side
    import sys_side

logger = logging.getLogger(__name__)


def _default_sweep_root() -> Path:
    return (Path(__file__).resolve().parent.parent / "output" / "sweeps").resolve()


def _default_figures_dir(base: Path, sweep_ts: str | None) -> Path:
    tag = sweep_ts or "latest"
    return (base / "figures" / tag).resolve()


def run_output(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    if input_path.is_dir():
        metas = collect_run_dirs(input_path)
        sweep_ts = args.sweep_ts or pick_latest_sweep_timestamp(metas)
        metas = filter_by_sweep_timestamp(metas, sweep_ts)
        run_dirs = [m.run_dir for m in metas]
        df = summarize_runs(run_dirs)
        if df.empty:
            logger.warning(f"No usable experiment results found: {input_path}")
            return
        output_dir = Path(args.output) if args.output else _default_figures_dir(input_path, sweep_ts)
    else:
        df = output_side.pd.read_csv(input_path)
        output_dir = Path(args.output) if args.output else input_path.parent / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_side.plot_malicious_ratio_vs_performance(df, output_dir)
    output_side.plot_attack_method_vs_malicious_benefit(df, output_dir)
    output_side.plot_dataset_robustness(df, output_dir)
    output_side.plot_total_requests_vs_performance(df, output_dir)
    logger.info(f"Output directory: {output_dir}")


def run_slo(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    csv_files = collect_results_csvs(input_path, args.sweep_ts)
    if not csv_files and input_path.is_dir():
        csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found: {input_path}")
        return

    sweep_ts = args.sweep_ts
    output_dir = Path(args.output) if args.output else _default_figures_dir(input_path, sweep_ts)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = [str(p) for p in csv_files]

    slo_side.plot_summary_by_payload(csv_paths, "normal", str(output_dir))
    slo_side.plot_summary_by_payload(csv_paths, "baseline", str(output_dir))
    slo_side.plot_summary_by_payload(csv_paths, "malicious", str(output_dir))
    logger.info(f"Output directory: {output_dir}")


def run_sys(args: argparse.Namespace) -> None:
    log_file = args.log_file
    run_dir = args.run_dir

    if not log_file:
        if run_dir:
            latest = find_latest_scheduler_log(Path(run_dir))
            if latest:
                log_file = str(latest)
        else:
            auto_run_dir = read_run_dir_from_env_or_file()
            if auto_run_dir:
                run_dir = str(auto_run_dir)
                latest = find_latest_scheduler_log(auto_run_dir)
                if latest:
                    log_file = str(latest)

    if not log_file:
        logger.error("Scheduler log file not found")
        return

    if not args.output:
        output_file = sys_side.generate_output_filename(log_file)
    else:
        output_path = Path(args.output)
        if output_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = str(output_path / "3_sys_side.png")
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = str(output_path)

    config = sys_side.load_config(args.config, run_dir=run_dir)
    sys_side.run_analysis(log_file, output_file, config)
    logger.info(f"Output file: {output_file}")


def run_all_for_run_dir(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = summarize_runs([run_dir])
    if not df.empty:
        output_side.plot_malicious_ratio_vs_performance(df, figures_dir)
        output_side.plot_attack_method_vs_malicious_benefit(df, figures_dir)
        output_side.plot_dataset_robustness(df, figures_dir)
        output_side.plot_total_requests_vs_performance(df, figures_dir)


def run_sys_per_experiment(args: argparse.Namespace) -> None:
    sweep_root = Path(args.run_dir).resolve()
    if not sweep_root.exists():
        logger.error(f"Directory does not exist: {sweep_root}")
        return

    log_file = args.log_file
    if not log_file:
        latest = find_latest_scheduler_log(sweep_root)
        if latest:
            log_file = str(latest)
        else:
            logger.error(f"Scheduler log file not found: {sweep_root}")
            return

    config = sys_side.load_config(args.config, run_dir=str(sweep_root))

    sys_side.generate_per_experiment_sys_plots(log_file, sweep_root, config)


def run_itl_per_experiment(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.postprocess import generate_per_experiment_itl_plots
    from visualize.paths import find_latest_scheduler_log

    sweep_root = Path(args.run_dir).resolve()
    if not sweep_root.exists():
        logger.error(f"Directory does not exist: {sweep_root}")
        return

    log_file = args.log_file
    if not log_file:
        latest = find_latest_scheduler_log(sweep_root)
        if latest:
            log_file = str(latest)
        else:
            logger.error(f"Scheduler log file not found: {sweep_root}")
            return

    generate_per_experiment_itl_plots(
        sweep_root=sweep_root,
        scheduler_log=Path(log_file),
        num_classes=args.num_classes,
        time_limit=args.time_limit,
        test_ratio=args.test_ratio,
    )


def cmd_sweep_summary(args: argparse.Namespace) -> None:
    from .sweep_summary import refresh_sweep_summary

    sweep_dir = Path(args.dir).resolve()
    if not sweep_dir.exists():
        logger.error(f"Directory does not exist: {sweep_dir}")
        return

    logger.info(f"Generating sweep summary table: {sweep_dir}")
    refresh_sweep_summary(sweep_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualization CLI")
    sub = parser.add_subparsers(dest="command")

    p_all = sub.add_parser("all", help="Run output+slo+sys")
    p_all.add_argument("--sweep-root", default=str(_default_sweep_root()))
    p_all.add_argument("--sweep-ts", default=None)
    p_all.add_argument("--run-dir", default=None)
    p_all.add_argument("--output", default=None)

    p_out = sub.add_parser("output", help="Run output-side plots")
    p_out.add_argument("--input", default=str(_default_sweep_root()))
    p_out.add_argument("--sweep-ts", default=None)
    p_out.add_argument("--output", default=None)

    p_slo = sub.add_parser("slo", help="Run slo-side plots")
    p_slo.add_argument("--input", default=str(_default_sweep_root()))
    p_slo.add_argument("--sweep-ts", default=None)
    p_slo.add_argument("--output", default=None)

    p_sys = sub.add_parser("sys", help="Run sys-side plots")
    p_sys.add_argument("--log-file", default=None)
    p_sys.add_argument("--run-dir", default=None)
    p_sys.add_argument("--output", default=None)
    p_sys.add_argument("--config", default=None)

    p_sys_per_exp = sub.add_parser("sys-per-exp", help="Generate an independent sys-side visualization for each experiment")
    p_sys_per_exp.add_argument("--run-dir", required=True, help="Sweep root directory")
    p_sys_per_exp.add_argument("--log-file", default=None, help="Path to the scheduler log file")
    p_sys_per_exp.add_argument("--config", default=None, help="Path to the configuration file")

    p_itl_per_exp = sub.add_parser("itl-per-exp", help="Generate an independent ITL visualization for each experiment")
    p_itl_per_exp.add_argument("--run-dir", required=True, help="Sweep root directory")
    p_itl_per_exp.add_argument("--log-file", default=None, help="Path to the scheduler log file")
    p_itl_per_exp.add_argument("--num-classes", type=int, default=10, help="Number of KV classes")
    p_itl_per_exp.add_argument("--time-limit", type=int, default=1200, help="Time limit (seconds)")
    p_itl_per_exp.add_argument("--test-ratio", type=float, default=0.2, help="Test set ratio")

    p_batch = sub.add_parser("batch", help="Recursively visualize each run directory")
    p_batch.add_argument("--root", required=True, help="Root directory to scan")
    p_batch.add_argument(
        "--sys-per-exp",
        action="store_true",
        help="Also generate per-experiment sys-side visualizations during batch processing (reusing parsed results)",
    )

    p_summary = sub.add_parser("sweep-summary", help="Generate the sweep summary table")
    p_summary.add_argument("--dir", required=True, help="Sweep output directory")

    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "output":
        run_output(args)
    elif args.command == "slo":
        run_slo(args)
    elif args.command == "sys":
        run_sys(args)
    elif args.command == "all":
        sweep_root = Path(args.sweep_root).resolve()
        sweep_ts = args.sweep_ts
        output_dir = Path(args.output) if args.output else _default_figures_dir(sweep_root, sweep_ts)
        output_dir.mkdir(parents=True, exist_ok=True)

        run_output(argparse.Namespace(input=str(sweep_root), sweep_ts=sweep_ts, output=str(output_dir)))
        run_slo(argparse.Namespace(input=str(sweep_root), sweep_ts=sweep_ts, output=str(output_dir)))
        run_sys(argparse.Namespace(log_file=None, run_dir=args.run_dir, output=str(output_dir), config=None))
    elif args.command == "batch":
        root = Path(args.root).resolve()
        if not root.exists():
            logger.error(f"Path does not exist: {root}")
            sys.exit(1)
        metas = collect_run_dirs(root)
        if not metas:
            logger.warning(f"No run directories found: {root}")
            return
        root_figures = root / "figures"
        root_figures.mkdir(parents=True, exist_ok=True)

        df = summarize_runs([m.run_dir for m in metas])
        if not df.empty:
            output_side.plot_malicious_ratio_vs_performance(df, root_figures)
            output_side.plot_attack_method_vs_malicious_benefit(df, root_figures)
            output_side.plot_dataset_robustness(df, root_figures)
            output_side.plot_total_requests_vs_performance(df, root_figures)

        run_slo(argparse.Namespace(input=str(root), sweep_ts=None, output=str(root_figures)))

        if args.sys_per_exp:
            log_file = None
            latest = find_latest_scheduler_log(root)
            if latest:
                log_file = str(latest)
            if not log_file:
                logger.error(f"Scheduler log file not found: {root}")
            else:
                config = sys_side.load_config(None, run_dir=str(root))
                output_file = str(root_figures / "3_sys_side.png")
                data_structures = sys_side.initialize_data_structures()
                data_structures = sys_side.parse_log_file(log_file, data_structures)
                sys_side.run_analysis(
                    log_file,
                    output_file,
                    config,
                    data_structures=data_structures,
                )
                sys_side.generate_per_experiment_sys_plots(
                    log_file,
                    root,
                    config,
                    data_structures=data_structures,
                )
        else:
            run_sys(argparse.Namespace(log_file=None, run_dir=str(root), output=str(root_figures), config=None))

        logger.info(f"Finished root directory visualization, covering {len(metas)} run directories")
    elif args.command == "sys-per-exp":
        run_sys_per_experiment(args)
    elif args.command == "itl-per-exp":
        run_itl_per_experiment(args)
    elif args.command == "sweep-summary":
        cmd_sweep_summary(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
