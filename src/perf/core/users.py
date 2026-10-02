
from __future__ import annotations

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Iterable, Union

from .strategy import Strategy, SimpleStrategy
from .results import RequestMetrics, TokenEvent
from .client import LLMClient
from .logger import MetricsLogger

import sys
from pathlib import Path
import random

_utils_dir = Path(__file__).parent.parent / "utils"
if str(_utils_dir) not in sys.path:
    sys.path.insert(0, str(_utils_dir))

_probe_dir = Path(__file__).parent.parent.parent / "probe"
if str(_probe_dir) not in sys.path:
    sys.path.insert(0, str(_probe_dir))

from utils.template import Prompts

logger = logging.getLogger("prober")


PromptInput = Union[Prompts, str]


def _normalize_prompts(
    prompts: Optional[Iterable[PromptInput]],
    fallback_text: str,
) -> List[Prompts]:
    if not prompts:
        return [Prompts(fallback_text)]
    normalized: List[Prompts] = []
    for item in prompts:
        if isinstance(item, Prompts):
            normalized.append(item)
        else:
            normalized.append(Prompts(str(item)))
    return normalized


def _prompt_texts(prompts: Iterable[PromptInput]) -> List[str]:
    texts: List[str] = []
    for item in prompts:
        if isinstance(item, Prompts):
            texts.append(item.prompt)
        else:
            texts.append(str(item))
    return [text for text in texts if text]


def _build_prompt_tiers(prompts: Iterable[Prompts]) -> Dict[str, List[str]]:
    tiers: Dict[str, List[Prompts]] = {}
    for prompt in prompts:
        tier = (prompt.tier or "high").lower()
        tiers.setdefault(tier, []).append(prompt)

    tier_texts: Dict[str, List[str]] = {}
    for tier, items in tiers.items():
        items.sort(key=lambda item: item.rank)
        tier_texts[tier] = [item.prompt for item in items if item.prompt]
    return {tier: texts for tier, texts in tier_texts.items() if texts}


class User:

    def __init__(
        self,
        user_id: int,
        user_type: str,
        strategy: Strategy,
        max_retries: int = 2,
        prompt_tiers: Optional[Dict[str, List[str]]] = None,
        prompts: Optional[List[Prompts]] = None,
        api_url: str | None = None,
    ):
        self.user_id = user_id
        self.user_type = user_type
        self.strategy = strategy
        self.results: list[RequestMetrics] = []
        self._client = LLMClient(user_id, user_type, api_url=api_url)
        self._max_retries = max_retries
        self._prompt_tiers = prompt_tiers or {}
        self.prompts: List[Prompts] = prompts or []
        self._cancelled = False

        self._current_request: Optional[RequestMetrics] = None

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    def cancel(self):
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def reset_cancel(self):
        self._cancelled = False

    def get_current_request(self) -> Optional[RequestMetrics]:
        return self._current_request

    def get_current_prompt_tokens(self) -> int:
        if self._current_request:
            return self._current_request.prompt_tokens
        return 0

    def get_current_completion_tokens(self) -> int:
        if self._current_request:
            return self._current_request.completion_tokens
        return 0

    def get_current_total_tokens(self) -> int:
        if self._current_request:
            return self._current_request.total_tokens
        return 0

    def get_current_latency(self) -> float:
        if self._current_request:
            return self._current_request.latency
        return 0.0

    def get_current_cost(
        self,
        input_cost_per_million: float = 0.15,
        output_cost_per_million: float = 0.60,
    ) -> float:
        if self._current_request:
            return self._current_request.cost(input_cost_per_million, output_cost_per_million)
        return 0.0

    def get_current_request_info(self) -> Dict[str, Any]:
        if self._current_request:
            m = self._current_request
            cost = self.get_current_cost()
            return {
                "prompt_tokens": m.prompt_tokens,
                "completion_tokens": m.completion_tokens,
                "total_tokens": m.total_tokens,
                "latency": m.latency,
                "cost_usd": cost,
                "is_active": True,
            }
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency": 0.0,
            "cost_usd": 0.0,
            "is_active": False,
        }

    def get_history_count(self) -> int:
        return len(self.results)

    def get_history_tokens(
        self,
        input_cost_per_million: float = 0.15,
        output_cost_per_million: float = 0.60,
    ) -> Dict[str, Any]:
        total_prompt = 0
        total_completion = 0
        total_cost = 0.0
        total_latency = 0.0
        success_count = 0
        failed_count = 0

        for m in self.results:
            if m.error:
                failed_count += 1
            else:
                success_count += 1
                total_prompt += m.prompt_tokens
                total_completion += m.completion_tokens
                total_cost += m.cost(input_cost_per_million, output_cost_per_million)
                total_latency += m.latency

        return {
            "total_requests": len(self.results),
            "success_requests": success_count,
            "failed_requests": failed_count,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_cost_usd": total_cost,
            "avg_latency": total_latency / success_count if success_count > 0 else 0.0,
        }

    def get_history_cost(
        self,
        input_cost_per_million: float = 0.15,
        output_cost_per_million: float = 0.60,
    ) -> float:
        return sum(
            m.cost(input_cost_per_million, output_cost_per_million)
            for m in self.results
            if not m.error
        )

    def get_all_history_info(
        self,
        input_cost_per_million: float = 0.15,
        output_cost_per_million: float = 0.60,
    ) -> Dict[str, Any]:
        current = self.get_current_request_info()
        history = self.get_history_tokens(input_cost_per_million, output_cost_per_million)

        if current["is_active"]:
            history["total_requests"] += 1
            history["prompt_tokens"] += current["prompt_tokens"]
            history["completion_tokens"] += current["completion_tokens"]
            history["total_tokens"] += current["total_tokens"]
            history["total_cost_usd"] += current["cost_usd"]

        return {
            "user_id": self.user_id,
            "user_type": self.user_type,
            "current_request": current,
            "history": history,
        }

    async def execute_request(
        self,
        prompt: str,
        api_url: str,
        model_name: str,
        scheduler_version: str = "v1",
        metrics_logger: Optional[MetricsLogger] = None,
    ) -> RequestMetrics:
        itl_callback = None
        if metrics_logger:
            itl_callback = metrics_logger.create_itl_callback()

        self._current_request = RequestMetrics(
            user_id=self.user_id,
            user_type=self.user_type,
            prompt=prompt,
        )

        metrics = await self._client.stream_request(
            prompt=prompt,
            api_url=api_url,
            model_name=model_name,
            scheduler_version=scheduler_version,
            max_retries=self._max_retries,
            itl_callback=itl_callback,
        )

        self._current_request = metrics

        if metrics_logger:
            metrics_logger.log_result(metrics)

        return metrics

    async def run(
        self,
        duration: int,
        metrics_logger: MetricsLogger,
        api_url: str,
        model_name: str,
        scheduler_version: str = "v1",
        max_startup_delay: float = 60.0,
    ) -> List[RequestMetrics]:
        import time

        start_time = time.time()
        end_time = start_time + duration
        # Note: do not use a fixed seed, so the random timings differ across experiments and better resemble real user behavior

        avg_interval = 3.0
        startup_delay = random.expovariate(1.0 / avg_interval)
        startup_delay = min(startup_delay, max_startup_delay)
        await asyncio.sleep(startup_delay)

        while time.time() < end_time:
            if self._cancelled:
                logger.info("[USER] user=%s received cancellation signal, stopping run", self.user_id)
                self.reset_cancel()
                break

            interval = self.strategy.get_interval(self.user_id)
            await asyncio.sleep(interval)

            if self._cancelled:
                break

            # Paper Sec. 4 / Algorithm 1 (Adaptive Back-off): when Delta_mem <= 0
            # the system is at or beyond saturation. Pause injection for t_wait
            # instead of dispatching, so the attacker's own requests are not
            # selected as preemption victims (self-preemption).
            backoff_wait = self.strategy.get_backoff_wait(self.user_id)
            if backoff_wait > 0:
                logger.info(
                    "[BACKOFF] user=%s system at/over saturation, pausing injection for %.2fs",
                    self.user_id,
                    backoff_wait,
                )
                await asyncio.sleep(backoff_wait)
                continue

            selected_tier = None
            if self._prompt_tiers:
                prompt = self.strategy.pick_prompt(self.user_id, self._prompt_tiers)
                for selector in self.strategy._prompt_selectors.values():
                    if hasattr(selector, "get_last_tier"):
                        selected_tier = selector.get_last_tier()
                        break
            elif self.prompts:
                prompt = random.choice(self.prompts).prompt
            elif hasattr(self, "_prompt") and self._prompt:
                prompt = self._prompt
            else:
                raise ValueError("No prompts available")

            if selected_tier:
                logger.info(
                    "[ATTACK] user=%s tier=%s prompt_len=%s",
                    self.user_id,
                    selected_tier,
                    len(prompt),
                )

            metrics = await self.execute_request(
                prompt=prompt,
                api_url=api_url,
                model_name=model_name,
                scheduler_version=scheduler_version,
                metrics_logger=metrics_logger,
            )

            if metrics.user_type == "malicious":
                metrics.tier = selected_tier

            self.results.append(metrics)

        return self.results

class NormalUser(User):

    def __init__(
        self,
        user_id: int,
        prompts: List[Prompts],
        lambda_value: float = 1.0,
        base_interval: float = 1.0,
        trace_path: str | None = None,
        trace_shuffle: bool = False,
    ):
        prompts = _normalize_prompts(prompts, "Default normal prompt")

        if trace_path:
            scheduler_config = {
                "type": "trace",
                "trace_path": trace_path,
                "repeat": True,
                "shuffle": trace_shuffle,
            }
        else:
            scheduler_config = {
                "type": "poisson",
                "base_interval": base_interval,
                "lambda_value": lambda_value,
            }

        config = {
            "prompt_selector": {
                "type": "file_list",
                "prompts": _prompt_texts(prompts),
                "shuffle": True,
                "repeat": True,
            },
            "interval_scheduler": scheduler_config,
        }
        strategy = SimpleStrategy(config)
        super().__init__(
            user_id=user_id,
            user_type="normal",
            strategy=strategy,
            prompts=prompts,
        )

    @classmethod
    def create(
        cls,
        user_id: int,
        prompts: List[Prompts],
        lambda_value: float = 1.0,
        base_interval: float = 1.0,
        trace_path: str | None = None,
        trace_shuffle: bool = False,
    ) -> "NormalUser":
        return cls(
            user_id=user_id,
            prompts=prompts,
            lambda_value=lambda_value,
            base_interval=base_interval,
            trace_path=trace_path,
            trace_shuffle=trace_shuffle,
        )


class WarmupUser(User):
    def __init__(
        self,
        user_id: int,
        prompt: str,
        base_interval: float = 0.1,
    ):
        config = {
            "prompt_selector": {
                "type": "file_list",
                "prompts": [prompt],
                "shuffle": False,
                "repeat": True,
            },
            "interval_scheduler": {
                "type": "fixed",
                "base_interval": base_interval,
                "jitter_range": 0.0,
            },
        }
        strategy = SimpleStrategy(config)
        super().__init__(
            user_id=user_id,
            user_type="warmup",
            strategy=strategy,
        )
        self._prompt = prompt

    @classmethod
    def create(cls, user_id: int, prompt: str, base_interval: float = 0.1) -> "WarmupUser":
        return cls(user_id=user_id, prompt=prompt, base_interval=base_interval)

class BaselineUser(User):

    @classmethod
    def create(
        cls,
        user_id: int,
        prompts: List[Prompts],
        base_interval: float = 1.0,
        jitter_range: float = 0.1,
    ) -> "BaselineUser":
        prompts = _normalize_prompts(prompts, "Default baseline prompt")
        config = {
            "prompt_selector": {
                "type": "file_list",
                "prompts": _prompt_texts(prompts),
                "shuffle": True,
                "repeat": True,
            },
            "interval_scheduler": {
                "type": "fixed",
                "base_interval": base_interval,
                "jitter_range": jitter_range,
            },
        }
        strategy = SimpleStrategy(config)
        return cls(
            user_id=user_id,
            user_type="baseline",
            strategy=strategy,
            prompt_tiers={},
            prompts=prompts,
        )


class FSAttackUser(User):

    @classmethod
    def create(
        cls,
        user_id: int,
        prompts: List[Prompts],
        scheduler_type: str = "burst",
        scheduler_config: Optional[Dict[str, Any]] = None,
        selector_config: Optional[Dict[str, Any]] = None,
    ) -> "FSAttackUser":
        scheduler_config = scheduler_config or {}
        selector_config = selector_config or {}

        # Arrival patterns are tuned to the baseline's average rate
        # (~60 RPM, i.e. mean gap 1.0 s) so that only the *strategy* differs,
        # not the offered traffic volume:
        #   fixed   : gap = 1.0 s                                  -> 60 RPM
        #   poisson : mean gap = base/lambda = 1.0/1.0 = 1.0 s     -> 60 RPM
        #   burst   : 0.2/0.4 + 0.8/1.6 = 1.0 req/s                -> 60 RPM
        #   wave    : mean rate = 1/sqrt(a^2-b^2) = 1.0 req/s      -> 60 RPM
        default_scheduler_configs = {
            "burst": {
                "base_interval": 1.6,
                "burst_interval": 0.4,
                "cycle_time": 15.0,
                "burst_time": 3.0,
            },
            "wave": {"base_interval": 1.25, "period": 10.0, "amplitude_ratio": 0.6},
            "poisson": {"base_interval": 1.0, "lambda_value": 1.0},
            "fixed": {"base_interval": 1.0, "jitter_range": 0.5},
        }

        # Paper Sec. 4 / Appendix Algorithm 1: the controller evaluates the
        # memory gap Delta_mem = C_sat - U_sys and selects
        #   Delta_mem >  delta_large               -> Fill       (P_High)
        #   delta_small < Delta_mem <= delta_large -> Transition (P_Mid)
        #   0 < Delta_mem <= delta_small           -> Squeeze    (P_Low)
        #   Delta_mem <= 0                         -> Back-off   (sleep t_wait)
        default_selector_config = {
            "c_sat": 97.5,  # saturation threshold C_sat (%)
            "delta_large": 47.5,  # Fill/Transition boundary at 50%
            "delta_small": 17.5,  # Transition/Squeeze boundary at 80%
            "t_wait": 1.0,  # Back-off pause (seconds)
            "max_state_age": 5.0,
        }

        final_scheduler_config = {
            **default_scheduler_configs.get(scheduler_type, {}),
            **scheduler_config,
            "type": scheduler_type,
        }
        # Allow selector_config to override type (for ablation studies: tiered instead of dynamic)
        selector_type = selector_config.pop("type", "dynamic")
        final_selector_config = {
            **default_selector_config,
            **selector_config,
            "type": selector_type,
        }

        prompt_tiers = _build_prompt_tiers(prompts)
        if not prompt_tiers:
            raise ValueError("FSAttackUser requires prompts with valid tiers")

        config = {
            "prompt_selector": final_selector_config,
            "interval_scheduler": final_scheduler_config,
        }

        strategy = SimpleStrategy(config)
        return cls(
            user_id=user_id,
            user_type="malicious",
            strategy=strategy,
            prompt_tiers=prompt_tiers,
            prompts=prompts,
        )


class ProberUser(User):

    @classmethod
    def create(
        cls,
        user_id: int,
        prompts: List[Prompts],
        interval: float = 0.1,
        model_dir: str | None = None,
        print_prediction: bool = True,
        window_min: int = 30,
        window_max: int = 50,
        prediction_logger: Optional[logging.Logger] = None,
    ) -> "ProberUser":
        prompts = _normalize_prompts(prompts, "System health check")

        config = {
            "prompt_selector": {
                "type": "file_list",
                "prompts": _prompt_texts(prompts),
                "shuffle": False,
                "repeat": True,
            },
            "interval_scheduler": {
                "type": "fixed",
                "base_interval": interval,
                "jitter_range": 0.0,
            },
        }
        predictor = None
        predictor_type = "none"

        try:
            from probe.core.inference import ITLPredictor
            predictor = ITLPredictor.get_cached(model_dir)
            predictor_type = "model"
            print(f"Loaded Probe model: {model_dir}")
        except Exception as e:
            raise ValueError(f"Failed to load Probe model: {e}")

        strategy = SimpleStrategy(config)
        return cls(
            user_id=user_id,
            user_type="prober",
            strategy=strategy,
            prompt_tiers={},
            prompts=prompts,
            predictor=predictor,
            predictor_type=predictor_type,
            print_prediction=print_prediction,
            window_min=window_min,
            window_max=window_max,
            prediction_logger=prediction_logger,
        )

    def __init__(
        self,
        user_id: int,
        user_type: str,
        strategy: Strategy,
        prompt_tiers: Optional[Dict[str, List[str]]] = None,
        prompts: Optional[List[Prompts]] = None,
        max_retries: int = 2,
        predictor=None,
        predictor_type: str = "none",
        print_prediction: bool = True,
        window_min: int = 30,
        window_max: int = 50,
        prediction_logger: Optional[logging.Logger] = None,
    ):
        super().__init__(
            user_id=user_id,
            user_type=user_type,
            strategy=strategy,
            prompt_tiers=prompt_tiers,
            max_retries=max_retries,
            prompts=prompts,
        )
        self._predictor = predictor
        self._predictor_type = predictor_type 
        self._print_prediction = print_prediction
        self._window_min = window_min
        self._window_max = window_max
        self._prediction_logger = prediction_logger
        from collections import deque
        self._itl_buf = deque(maxlen=window_max)
        self._since_last_pred = 0
        self._step_size = 15

    async def execute_request(
        self,
        prompt: str,
        api_url: str,
        model_name: str,
        scheduler_version: str = "v1",
        metrics_logger: Optional[MetricsLogger] = None,
    ) -> RequestMetrics:
        req_num = self._client.request_count + 1

        self._itl_buf.clear()
        self._since_last_pred = 0
        self._token_count = 0

        logger_callback = metrics_logger.create_itl_callback() if metrics_logger else None

        def combined_callback(event: TokenEvent):
            self._token_count += 1

            if logger_callback:
                logger_callback(event)

            itl_ms = event.itl_ms
            self._itl_buf.append(itl_ms)
            self._since_last_pred += 1

            buf_len = len(self._itl_buf)

            if self._predictor:
                try:
                    if buf_len >= self._window_min and self._since_last_pred >= self._step_size:
                        window = list(self._itl_buf)
                        pred_class = self._predictor.predict_class(window)
                        pred_range = self._predictor._class_to_range(pred_class)
                        logger.info("Predicted class: %s, range: %s", pred_class,pred_range)

                        from .state_manager import get_state_manager
                        state_manager = get_state_manager()

                        kv_usage_percent = pred_class * 10.0
                        state_manager.update(
                            kv_usage_class=pred_class,
                            kv_usage_percent=kv_usage_percent,
                            itl_window=window.copy(),
                        )

                        if self._prediction_logger:
                            try:
                                window_vals = [round(x, 4) for x in window]
                                self._prediction_logger.info(
                                    "",
                                    extra={
                                        "time": event.wall_time,
                                        "predict_class": pred_class,
                                        "window": json.dumps(window_vals),
                                    },
                                )
                            except Exception as e:
                                logger.error(
                                    "[PROBE] user=%s req=%s PREDICT_LOG_FAILED: %s",
                                    self.user_id,
                                    req_num,
                                    e,
                                )
                        
                        self._since_last_pred = 0
                except Exception as e:
                    logger.error("[PROBE] user=%s req=%s PREDICT_FAILED: %s", self.user_id, req_num, e)
                    if self._print_prediction:
                        print(f"[PROBE] Prediction failed: {e}")

        metrics = await self._client.stream_request(
            prompt=prompt,
            api_url=api_url,
            model_name=model_name,
            scheduler_version=scheduler_version,
            max_retries=self._max_retries,
            itl_callback=combined_callback,
        )

        if metrics_logger:
            metrics_logger.log_result(metrics)

        return metrics


def create_user(
    user_type: str,
    user_id: int,
    **kwargs,
) -> User:
    user_classes = {
        "normal": NormalUser,
        "baseline": BaselineUser,
        "fs_attack": FSAttackUser,
        "prober": ProberUser,
    }

    if user_type not in user_classes:
        raise ValueError(
            f"Unknown user type: {user_type}. Available types: {list(user_classes.keys())}"
        )

    return user_classes[user_type].create(user_id=user_id, **kwargs)


__all__ = [
    "User",
    "NormalUser",
    "BaselineUser",
    "FSAttackUser",
    "ProberUser",
    "create_user",
]
