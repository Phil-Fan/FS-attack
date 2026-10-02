#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import logging
import pandas as pd
import yaml
import os
import json

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.config import ConfigLoader, load_config, list_sweeps, get_server, get_experiment
from core.logger import MetricsLogger, setup_logging, get_logger
from core.users import FSAttackUser, NormalUser, ProberUser, WarmupUser, BaselineUser
from core.sweep_iterator import load_sweep
from utils.load_data import load_prompts, Prompts


def _load_prompts_from_config(
    dataset_cfg,
    *,
    default_template: str = "plain",
) -> List[Prompts]:
    if not dataset_cfg:
        return []
    if isinstance(dataset_cfg, list):
        prompts: List[Prompts] = []
        for item in dataset_cfg:
            prompts.extend(
                _load_prompts_from_config(item, default_template=default_template)
            )
        return prompts
    if isinstance(dataset_cfg, dict):
        path = dataset_cfg.get("path") or dataset_cfg.get("file")
        if not path:
            return []
        template = dataset_cfg.get("template", default_template)
        max_prompts = dataset_cfg.get("max_prompts")
        tier = dataset_cfg.get("tier")
        prompts = load_prompts(path, template, max_prompts)
        if tier and prompts:
            for p in prompts:
                p.tier = tier
        return prompts
    if isinstance(dataset_cfg, str):
        return load_prompts(dataset_cfg, default_template)
    return []


async def resolve_server_model(server_cfg: Dict) -> str:
    if server_cfg.get("fetch_model"):
        try:
            import httpx
            from urllib.parse import urlparse

            api_url = server_cfg.get("api_url", "")
            base_url = urlparse(api_url).scheme + "://" + urlparse(api_url).netloc
            if "/v1/" in api_url:
                base_url = api_url.split("/v1/", 1)[0]

            async with httpx.AsyncClient(timeout=5.0) as client:
                model_name = (await client.get(f"{base_url}/v1/models")).json()["data"][0]["id"]
                print(f"API model: {model_name}")
                return model_name
        except Exception as e:
            print(f"Failed to fetch from API, using config: {e}")

    model_path = server_cfg.get("model_path")
    if not model_path:
        raise ValueError("Please set server.model_path or enable server.fetch_model")
    return model_path


@dataclass
class ExperimentResult:
    total: int
    success: int
    failed: int
    avg_latency: float
    normal_count: int
    normal_avg: float
    malicious_count: int
    malicious_avg: float
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0
    input_cost_per_million: float = 0.15
    output_cost_per_million: float = 0.60


class Experiment:

    def __init__(self, cfg: Dict, log_dir: Path):
        self.cfg = cfg
        self.meta = cfg.get("_meta", {})
        self.users: List = []
        self.log_dir = log_dir
        self._api_url: str = ""

        log_file = log_dir / "experiment.log"
        setup_logging(
            log_file=str(log_file),
            console_level=logging.INFO,
            file_level=logging.DEBUG,
            logger_names=["prober", "attack", "sweep"],
        )

        self.metrics_logger = MetricsLogger.create_from_paths(
            result_log_path=str(log_dir / "results.csv"),
            itl_log_path=str(log_dir / "itl.csv"),
        )
        self.prober_pred_logger = get_logger(
            "prober_predictions",
            str(log_dir / "prober_predictions.csv"),
            is_csv=True,
            fieldnames=["time", "predict_class", "window"],
        )

    async def _cleanup_users(self) -> None:
        for user in self.users:
            try:
                await user.close()
            except Exception:
                pass

    def _create_users(self):
        users_cfg = self.cfg.get("users", {})
        datasets_cfg = self.cfg.get("datasets", {})
        prober_cfg = self.cfg.get("prober", {})
        server_cfg = get_server(self.cfg)

        total = users_cfg.get("total", 20)
        ratios = users_cfg.get("ratios", {})
        seed = int(os.environ.get("EXP_SEED", 41))
        random.seed(seed)

        # Legacy mode: special_ratios mode, where ratios holds ratios
        max_ratio_value = max(ratios.values()) if ratios else 0
        use_fixed_count_mode = max_ratio_value > 1

        baseline_count = int(total * ratios.get("baseline", 0)) if not use_fixed_count_mode else ratios.get("baseline", 0)
        attack_count = int(total * ratios.get("fs_attack", 0)) if not use_fixed_count_mode else ratios.get("fs_attack", 0)
        prober_count = ratios.get("prober", 0)
        warmup_count = ratios.get("warmup", 0)
        normal_count = total - baseline_count - attack_count - prober_count - warmup_count

        available_ids = set(range(1, total + 1))
        self.users = []

        def get_random_id() -> int:
            uid = random.choice(list(available_ids))
            available_ids.remove(uid)
            return uid

        if warmup_count > 0:
            ds_cfg = datasets_cfg.get("warmup", datasets_cfg.get("normal", {}))
            warmup_prompts: List[Prompts] = _load_prompts_from_config(
                ds_cfg, default_template="plain"
            )
            for _ in range(warmup_count):
                user = WarmupUser.create(
                    user_id=get_random_id(),
                    prompt=warmup_prompts[random.randint(0, len(warmup_prompts) - 1)].prompt if warmup_prompts else "Hello",
                )
                user._client._init_client(server_cfg["api_url"])
                self.users.append(user)

        if baseline_count > 0:
            ds_cfg = datasets_cfg.get("baseline", datasets_cfg.get("normal", {}))
            prompts = _load_prompts_from_config(ds_cfg, default_template="plain")
            for _ in range(baseline_count):
                user = BaselineUser.create(
                    user_id=get_random_id(),
                    prompts=prompts,
                    base_interval=users_cfg.get("baseline_base_interval", 1.0),
                    jitter_range=users_cfg.get("baseline_jitter", 0.5),
                )
                user._client._init_client(server_cfg["api_url"])
                self.users.append(user)

        if attack_count > 0:
            fs_attack_cfg = datasets_cfg.get("fs_attack", datasets_cfg.get("normal", {}))
            if isinstance(fs_attack_cfg, dict) and any(tier in fs_attack_cfg for tier in ("high", "mid", "low")):
                prompts: List[Prompts] = []
                for tier in ("high", "mid", "low"):
                    tier_cfg = fs_attack_cfg.get(tier, [])
                    if tier_cfg:
                        tier_prompts = _load_prompts_from_config(tier_cfg, default_template="fs_attack")
                        if tier_prompts:
                            prompts.extend(tier_prompts)
                            print(f"  [FS-ATTACK] tier={tier}: loaded {len(tier_prompts)} prompts")
                print(f"  [FS-ATTACK] loaded {len(prompts)} prompts in total (tiered)")
            else:
                # Legacy format: a single dataset or a list
                prompts = _load_prompts_from_config(fs_attack_cfg, default_template="fs_attack")
                print(f"  [FS-ATTACK] loaded {len(prompts)} prompts")

            scheduler_type = users_cfg.get("malicious_pattern", "burst")
            selector_config = self.cfg.get("fs_attack_selector", {})
            for _ in range(attack_count):
                user = FSAttackUser.create(
                    user_id=get_random_id(),
                    scheduler_type=scheduler_type,
                    prompts=prompts,
                    selector_config=selector_config,
                )
                user._client._init_client(server_cfg["api_url"])
                self.users.append(user)

        if normal_count > 0:
            ds_cfg = datasets_cfg.get("normal", {})
            normal_prompts: List[Prompts] = _load_prompts_from_config(
                ds_cfg, default_template="plain"
            )

            normal_pattern = users_cfg.get("normal_pattern", "poisson")
            normal_base_interval = users_cfg.get("normal_base_interval", 1.0)
            normal_lambda = users_cfg.get("normal_lambda", 1.0)
            normal_trace_path = users_cfg.get("normal_trace_path")
            normal_trace_shuffle = users_cfg.get("normal_trace_shuffle", False)

            use_trace = normal_pattern == "trace" and normal_trace_path

            for _ in range(normal_count):
                user = NormalUser.create(
                    user_id=get_random_id(),
                    prompts=normal_prompts,
                    lambda_value=normal_lambda if not use_trace else 1.0,
                    base_interval=normal_base_interval if not use_trace else 1.0,
                    trace_path=normal_trace_path if use_trace else None,
                    trace_shuffle=normal_trace_shuffle,
                )
                user._client._init_client(server_cfg["api_url"])
                self.users.append(user)

        if prober_count > 0:
            ds_cfg = datasets_cfg.get("prober", {})
            prober_prompts: List[Prompts] = _load_prompts_from_config(
                ds_cfg, default_template="plain"
            )
            for _ in range(prober_count):
                prober = ProberUser.create(
                    user_id=get_random_id(),
                    prompts=prober_prompts,
                    interval=float(prober_cfg.get("interval", 0.1)),
                    model_dir=prober_cfg.get("model_dir"),
                    print_prediction=bool(prober_cfg.get("print", True)),
                    window_min=int(prober_cfg.get("window_min", 30)),
                    window_max=int(prober_cfg.get("window_max", 50)),
                    prediction_logger=self.prober_pred_logger,
                )
                prober._client._init_client(server_cfg["api_url"])
                self.users.append(prober)

        print(f"User distribution:")
        if warmup_count > 0:
            print(f"  - warmup: {warmup_count}")
        if baseline_count > 0:
            print(f"  - baseline: {baseline_count}")
        if attack_count > 0:
            print(f"  - fs_attack: {attack_count}")
        if normal_count > 0:
            print(f"  - normal: {normal_count}")
        if prober_count > 0:
            print(f"  - prober: {prober_count}")
        print(f"  Total: {len(self.users)} users")

    async def _warmup(self):
        print("\nWarming up...")
        import random as random_module

        exp_cfg = get_experiment(self.cfg)
        server_cfg = get_server(self.cfg)
        self._api_url = server_cfg["api_url"]

        model_name = await resolve_server_model(server_cfg)

        warmup_users = exp_cfg.get("warmup_users", 20)
        warmup_count = exp_cfg.get("warmup_count", 30)
        warmup_prompt = exp_cfg.get("warmup_prompt", "Hello")

        warmup_prompts = [
            warmup_prompt,
            "Explain quantum computing in simple terms.",
            "Write a short poem about spring.",
            "What is the capital of France?",
            "How do plants perform photosynthesis?",
        ]

        print(f"  Parallel users: {warmup_users}, serial requests per user: {warmup_count}")

        async def warmup_user_task(user_idx: int) -> tuple:
            warmup_user = WarmupUser.create(
                user_id=-user_idx - 1,  # Use negative IDs to avoid clashing with normal users
                prompt=random_module.choice(warmup_prompts),
            )
            warmup_user._client._init_client(server_cfg["api_url"])

            success = 0
            for j in range(warmup_count):
                try:
                    metrics = await warmup_user.execute_request(
                        prompt=random_module.choice(warmup_prompts),
                        api_url=server_cfg["api_url"],
                        model_name=model_name,
                        scheduler_version=server_cfg["scheduler_version"],
                        metrics_logger=None,
                    )
                    if not metrics.error:
                        success += 1
                except Exception:
                    pass

            await warmup_user.close()
            return success

        user_tasks = [warmup_user_task(i) for i in range(warmup_users)]
        user_results = await asyncio.gather(*user_tasks)

        total_success = sum(user_results)
        total_requests = warmup_users * warmup_count

        print(f"  Done: {total_success}/{total_requests} succeeded ({100*total_success/total_requests:.1f}%)")
        print("Warmup complete\n")

    async def run(self) -> ExperimentResult:
        exp_cfg = get_experiment(self.cfg)
        server_cfg = get_server(self.cfg)

        model_name = await resolve_server_model(server_cfg)

        print("=" * 50)
        print("Starting experiment")
        print("=" * 50)
        print(f"Config: {self.meta.get('config_name', 'unknown')}")
        print(f"Duration: {exp_cfg.get('duration', 60)}s")
        print(f"Model: {model_name}")
        print(f"API: {server_cfg['api_url']}")
        print("=" * 50 + "\n")

        self._create_users()

        await self._warmup()

        start_time = datetime.now()

        duration_seconds = exp_cfg.get("duration", 60)
        expected_end_time = start_time + pd.Timedelta(seconds=duration_seconds)

        meta_file = self.log_dir / "experiment_meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "start_time": start_time.isoformat(),
                "expected_end_time": expected_end_time.isoformat(),
                "configured_duration_seconds": duration_seconds,
            }, f, indent=2)
        print(f"Experiment metadata saved: {meta_file}")

        print("Running...")
        max_startup_delay = exp_cfg.get("max_startup_delay", 60.0)
        tasks = [
            user.run(
                duration=exp_cfg.get("duration", 60),
                metrics_logger=self.metrics_logger,
                api_url=server_cfg["api_url"],
                model_name=model_name,
                scheduler_version=server_cfg["scheduler_version"],
                max_startup_delay=max_startup_delay,
            )
            for user in self.users
        ]

        results_list = await asyncio.gather(*tasks)
        all_results = [r for results in results_list for r in results]

        actual_end_time = datetime.now()

        meta_file = self.log_dir / "experiment_meta.json"
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["actual_end_time"] = actual_end_time.isoformat()
        meta["actual_duration_seconds"] = (actual_end_time - start_time).total_seconds()
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Experiment metadata updated: {meta_file}")

        normal = [r for r in all_results if r.user_type == "normal"]
        malicious = [r for r in all_results if r.user_type == "malicious"]
        success = [r for r in all_results if not r.error]

        billing_cfg = self.cfg.get("billing", {})
        billing_enabled = billing_cfg.get("enabled", True)
        input_cost = billing_cfg.get("input_cost_per_million", 0.15)
        output_cost = billing_cfg.get("output_cost_per_million", 0.60)

        total_prompt_tokens = sum(r.prompt_tokens for r in success)
        total_completion_tokens = sum(r.completion_tokens for r in success)
        total_cost = (total_prompt_tokens / 1000000) * input_cost + (total_completion_tokens / 1000000) * output_cost

        result = ExperimentResult(
            total=len(all_results),
            success=len(success),
            failed=len(all_results) - len(success),
            avg_latency=sum(r.latency for r in all_results) / len(all_results) if all_results else 0,
            normal_count=len(normal),
            normal_avg=sum(r.latency for r in normal) / len(normal) if normal else 0,
            malicious_count=len(malicious),
            malicious_avg=sum(r.latency for r in malicious) / len(malicious) if malicious else 0,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_cost=total_cost,
            input_cost_per_million=input_cost,
            output_cost_per_million=output_cost,
        )

        print("\n" + "=" * 50)
        print("Results")
        print("=" * 50)
        print(f"Total: {result.total} | Success: {result.success} | Failed: {result.failed}")
        print(f"Average latency: {result.avg_latency:.2f}s")
        print(f"Normal users: {result.normal_count} requests, avg {result.normal_avg:.2f}s")
        print(f"Malicious users: {result.malicious_count} requests, avg {result.malicious_avg:.2f}s")
        if billing_enabled:
            print("-" * 50)
            print(f"Cost summary:")
            print(f"   Input tokens: {total_prompt_tokens:,} (${input_cost}/M)")
            print(f"   Output tokens: {total_completion_tokens:,} (${output_cost}/M)")
            print(f"   Total cost: ${total_cost:.4f}")
        print("=" * 50 + "\n")

        await self._cleanup_users()

        return result


def _write_config_snapshot(cfg: Dict, run_dir: Path) -> None:
    config_path = run_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def _prepare_run_dir(cfg: Dict, run_tag: str, output_dir: str | None = None) -> Path:
    exp_cfg = get_experiment(cfg)

    if output_dir:
        run_dir = Path(output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    env_run_dir = os.environ.get("RUN_OUTPUT_DIR")
    if env_run_dir:
        run_dir = Path(env_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    output_root = Path(exp_cfg.get("output_dir", "output"))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{timestamp}__{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


async def run_single(config_name: str, output_dir: str | None = None) -> int:
    try:
        cfg = load_config(config_name)
        run_tag = cfg.get("_meta", {}).get("config_name", "experiment")
        run_dir = _prepare_run_dir(cfg, run_tag, output_dir)
        cfg.setdefault("_meta", {})["config_dir"] = str(run_dir)
        cfg.setdefault("experiment", {})["output_dir"] = str(run_dir)
        _write_config_snapshot(cfg, run_dir)

        log_dir = run_dir / "logs"
        exp = Experiment(cfg, log_dir)
        await exp.run()
        return 0
    except FileNotFoundError as e:
        print(f"{e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


async def run_sweep_from_yaml(sweep_name: str, loader: ConfigLoader, dry_run: bool = False, output_dir: str | None = None) -> int:
    sweep = load_sweep(sweep_name, loader)
    env_run_root = os.environ.get("RUN_OUTPUT_DIR")
    if env_run_root:
        run_root = Path(env_run_root)
    elif output_dir:
        run_root = Path(output_dir)
    else:
        run_root = Path(sweep.output_dir)
        if not dry_run:
            raise ValueError("The RUN_OUTPUT_DIR environment variable is not set")

    base_cfg = loader._load_base()

    # Prefer base_overrides.total, otherwise use base_config.users.total
    if sweep.base_overrides and "total" in sweep.base_overrides:
        total_users = sweep.base_overrides["total"]
    else:
        total_users = base_cfg.get("users", {}).get("total", 20)

    # Ensure the directory exists
    if not dry_run:
        run_root.mkdir(parents=True, exist_ok=True)

    logs_dir = run_root / "logs"
    if not dry_run:
        logs_dir.mkdir(parents=True, exist_ok=True)

    output_dir = run_root

    total = sweep.total_count()

    print("=" * 50)
    print(f"SWEEP experiment: {sweep_name}")
    print("=" * 50)
    print(f"Run directory: {run_root}")
    print(f"Durations: {sweep.durations}")
    print(f"Special user ratios: {sweep.special_ratios}")
    print(f"Prober counts: {sweep.prober_counts}")
    print(f"Repeats: {sweep.repeats}")
    print(f"Total combinations: {total}")
    print("=" * 50 + "\n")

    if not dry_run:
        try:
            from visualize.sweep_summary import initialize_sweep_table

            rows_data = []
            for combo in sweep:
                # 2. Legacy mode: compute from ratios
                if combo.normal_count is not None and combo.malicious_count is not None:
                    malicious_count = combo.malicious_count
                    normal_count = combo.normal_count
                    total_users = normal_count + malicious_count + combo.prober_count
                else:
                    malicious_count = int(total_users * combo.special_ratio) if combo.special_type else 0
                    normal_count = total_users - malicious_count - combo.prober_count

                rows_data.append({
                    "run_id": combo.combo_id,
                    "normal_user_count": normal_count,
                    "normal_dataset": combo.normal_dataset,
                    "malicious_user_count": malicious_count,
                    "attack_method": combo.special_dataset
                })

            print(f"Preparing to initialize the table with {len(rows_data)} sub-experiments")
            print(f"   Output directory: {output_dir}")

            initialize_sweep_table(output_dir, rows_data)
            print("Table initialization complete")
        except Exception as e:
            import traceback
            print(f"Table initialization failed: {e}")
            traceback.print_exc()

    run_count = 0
    for combo in sweep:
        run_count += 1
        print(f"\n[{run_count}/{total}] {combo.combo_id}")

        if dry_run:
            def get_dataset_tier(ds_name: str) -> str:
                try:
                    datasets_cfg = loader._load_datasets()
                    parts = ds_name.split(".")
                    if len(parts) >= 2:
                        category, name = parts[0], parts[1]
                        category_cfg = datasets_cfg.get(category, {})
                        if isinstance(category_cfg, dict):
                            ds_cfg = category_cfg.get(name, {})
                            if isinstance(ds_cfg, dict):
                                return ds_cfg.get("tier", "N/A")
                    return "N/A"
                except Exception:
                    return "N/A"

            def get_tiered_datasets_info() -> str:
                if combo.special_datasets_tiered:
                    parts = []
                    for tier in ("high", "mid", "low"):
                        ds_list = combo.special_datasets_tiered.get(tier, [])
                        if ds_list:
                            ds_info = []
                            for ds in ds_list:
                                ds_tier = get_dataset_tier(ds)
                                ds_info.append(f"{ds}[{ds_tier}]")
                            parts.append(f"{tier}:[{','.join(ds_info)}]")
                    return " | ".join(parts)
                return None

            print("  [DRY-RUN]")
            print(f"    Output directory: {combo.combo_id}/")
            print(f"    Duration: {combo.duration}s")
            normal_tier = get_dataset_tier(combo.normal_dataset)
            print(f"    Normal dataset: {combo.normal_dataset} [tier: {normal_tier}]")
            print(f"    Attack type: {combo.special_type}")
            tiered_info = get_tiered_datasets_info()
            if tiered_info:
                print(f"    Attack dataset: (tiered)")
                for tier_line in tiered_info.split(" | "):
                    print(f"      {tier_line}")
            elif isinstance(combo.special_dataset, list):
                for ds in combo.special_dataset:
                    ds_tier = get_dataset_tier(ds)
                    print(f"    Attack dataset: {ds} [tier: {ds_tier}]")
            else:
                ds_tier = get_dataset_tier(combo.special_dataset)
                print(f"    Attack dataset: {combo.special_dataset} [tier: {ds_tier}]")
            print("    User composition:")
            if combo.normal_count is not None and combo.malicious_count is not None:
                combo_total = combo.normal_count + combo.malicious_count + combo.prober_count
                print(f"      - Malicious users: {combo.malicious_count}")
                print(f"      - Normal users: {combo.normal_count}")
                print(f"      - Probers: {combo.prober_count}")
                print(f"      - Total: {combo_total}")
                print(f"      - Malicious share: {combo.special_ratio:.2%}")
            else:
                prober = combo.prober_count
                malicious = int(total_users * combo.special_ratio)
                normal = total_users - malicious - prober
                print(f"      - Malicious users: {malicious} ({combo.special_ratio:.0%})")
                print(f"      - Normal users: {normal}")
                print(f"      - Probers: {prober}")
                print(f"      - Total: {total_users}")
            continue

        run_dir = output_dir / combo.combo_id

        cfg = loader._load_base()

        cfg = combo.to_config(cfg)

        if sweep.base_overrides:
            for key, value in sweep.base_overrides.items():
                if isinstance(value, dict):
                    if key == "users":
                        cfg.setdefault("users", {}).update(value)
                    elif key == "prober":
                        cfg.setdefault("prober", {}).update(value)
                    elif key == "experiment":
                        cfg.setdefault("experiment", {}).update(value)
                    elif key == "datasets":
                        cfg.setdefault("datasets", {}).update(value)
                    else:
                        cfg.setdefault(key, {}).update(value)
                else:
                    if key in ("server",):
                        cfg[key] = value
                    else:
                        cfg.setdefault("users", {})[key] = value

        cfg.setdefault("experiment", {})["output_dir"] = str(run_dir)
        cfg.setdefault("_meta", {})["config_dir"] = str(run_dir)

        cfg = loader._expand_config(cfg, run_dir)

        run_dir.mkdir(parents=True, exist_ok=True)
        _write_config_snapshot(cfg, run_dir)

        print(f"  Configuration saved: {run_dir / 'config.yaml'}")

        log_dir = run_dir / "logs"
        exp = Experiment(cfg, log_dir)
        await exp.run()

        try:
            from visualize.sweep_summary import update_sweep_tables
            update_sweep_tables(run_dir)
        except Exception as e:
            print(f"Failed to update the table: {e}")

        if run_count < total:
            print("Waiting 20 seconds...")
            await asyncio.sleep(20)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Latency attack experiments")
    parser.add_argument("--config", help="Path to the configuration file")
    parser.add_argument("--sweep", nargs="?", const="benchmark", help="Run a Sweep (specify a name or use the default)")
    parser.add_argument("--list-sweeps", action="store_true", help="List all available Sweeps")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dir", "-d", help="Specify the output directory directly (takes priority over the environment variable)")
    return asyncio.run(_main(parser.parse_args()))


async def _main(args: argparse.Namespace) -> int:
    if args.list_sweeps:
        sweeps = list_sweeps()
        print("Available Sweep configurations:")
        for name, info in sweeps.items():
            status = "enabled" if info["enabled"] else "disabled"
            print(f"  {status} {name}")
        return 0

    if args.sweep:
        loader = ConfigLoader()
        return await run_sweep_from_yaml(args.sweep, loader, dry_run=args.dry_run, output_dir=args.dir)

    if args.config:
        return await run_single(args.config, output_dir=args.dir)

    print("Latency Attack: unified entry point for latency attack experiments")
    print("\nAvailable commands:")
    print("  python attack_main.py --config <name>              # Run an experiment")
    print("  python attack_main.py --sweep                # Run the default Sweep")
    print("  python attack_main.py --sweep benchmark      # Run a specific Sweep")
    print("  python attack_main.py --list-sweeps          # List all Sweeps")
    print("  python attack_main.py --config <name> --dir <path>  # Specify the output directory")
    return 0


if __name__ == "__main__":
    sys.exit(main())
