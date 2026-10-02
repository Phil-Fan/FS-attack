
from __future__ import annotations

import time
import os
from datetime import datetime
from typing import Optional, Dict, Any, AsyncIterator, Tuple, List

from openai import AsyncOpenAI

from .results import RequestMetrics, TokenEvent

ENABLE_THINKING = os.environ.get("ENABLE_THINKING", False)

class LLMClient:
    def _build_extra_body(
        self,
        scheduler_version: str,
        metadata: Dict[str, Any],
        model_name: str,
    ) -> Dict[str, Any]:
        extra_body: Dict[str, Any] = {
            "chat_template_kwargs": {
                "enable_thinking": False if not ENABLE_THINKING else True
            },
        }

        backend = os.environ.get("PERF_BACKEND", "").strip().lower()
        is_sglang = (
            scheduler_version.lower() == "sglang"
            or backend == "sglang"
            or "sglang" in model_name.lower()
        )

        if is_sglang:
            extra_body["custom_params"] = metadata
            return extra_body

        args_key = "extra_args" if scheduler_version == "v0" else "vllm_xargs"
        extra_body[args_key] = metadata
        return extra_body

    def __init__(self, user_id: int, user_type: str, api_url: str | None = None):
        self.user_id = user_id
        self.user_type = user_type
        self._request_count = 0
        self._client: AsyncOpenAI | None = None
        self._base_url: str | None = None

        if api_url:
            self._init_client(api_url)

    def _init_client(self, api_url: str) -> None:
        if self._client is not None:
            return

        self._base_url = self._build_base_url(api_url)
        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        self._client = AsyncOpenAI(api_key=api_key, base_url=self._base_url)

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def client(self) -> AsyncOpenAI | None:
        return self._client

    def _ensure_client(self, api_url: str) -> None:
        if self._client is None:
            self._init_client(api_url)

    async def close(self) -> None:
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None

    def _build_base_url(self, api_url: str) -> str:
        if "/v1/" in api_url:
            prefix = api_url.split("/v1/", 1)[0]
            return f"{prefix}/v1"
        elif api_url.rstrip("/").endswith("/v1"):
            return api_url.rstrip("/")
        return api_url.rstrip("/")

    async def stream_request(
        self,
        prompt: str,
        api_url: str,
        model_name: str,
        scheduler_version: str = "v1",
        max_retries: int = 2,
        itl_callback: Optional[[TokenEvent], None] = None,
    ) -> RequestMetrics:
        self._request_count += 1
        num_run = self._request_count

        # Ensure the client is initialized (lazy initialization)
        self._ensure_client(api_url)

        client = self._client
        assert client is not None, "Client should be initialized"

        metadata = {
            "type": self.user_type,
            "user_id": str(self.user_id),
            "experiment_name": "latency_attack_test",
            "timestamp": int(time.time()),
        }

        for attempt in range(max_retries):
            start_time = time.perf_counter()

            try:
                print(f"ENABLE_THINKING: {ENABLE_THINKING}")
                stream = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body=self._build_extra_body(
                        scheduler_version=scheduler_version,
                        metadata=metadata,
                        model_name=model_name,
                    ),
                )

                metrics = await self._process_stream(
                    stream=stream,
                    user_id=self.user_id,
                    user_type=self.user_type,
                    num_run=num_run,
                    prompt=prompt,
                    start_time=start_time,
                    itl_callback=itl_callback,
                )

                if metrics.error:
                    if attempt < max_retries - 1:
                        continue

                return metrics

            except Exception as e:
                latency = time.perf_counter() - start_time
                status_code = int(getattr(e, "status_code", 0) or 0)

                if attempt < max_retries - 1:
                    continue

                return RequestMetrics(
                    user_id=self.user_id,
                    user_type=self.user_type,
                    num_run=num_run,
                    prompt=prompt,
                    latency=latency,
                    status=status_code,
                    error=f"{type(e).__name__}: {e}",
                )

        return RequestMetrics(
            user_id=self.user_id,
            user_type=self.user_type,
            num_run=num_run,
            prompt=prompt,
            latency=0.0,
            error="Exceeded retry attempts",
        )

    async def _process_stream(
        self,
        stream: AsyncIterator,
        user_id: int,
        user_type: str,
        num_run: int,
        prompt: str,
        start_time: float,
        itl_callback: Optional[[TokenEvent], None] = None,
    ) -> RequestMetrics:
        generated_text = ""
        ttft = 0.0
        itl: List[float] = []
        most_recent_timestamp = start_time
        first_chunk_received = False
        token_idx = 0

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        async for event in stream:
            timestamp = time.perf_counter()
            current_time = time.time()
            wall_time = datetime.fromtimestamp(current_time).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]

            if getattr(event, "choices", None):
                delta = event.choices[0].delta
                text = getattr(delta, "content", None) or ""

                if text:
                    if not first_chunk_received:
                        ttft = timestamp - start_time
                        first_chunk_received = True
                    else:
                        itl_value = timestamp - most_recent_timestamp
                        itl.append(itl_value)
                        token_idx += 1

                        if itl_callback:
                            token_event = TokenEvent(
                                user_id=user_id,
                                user_type=user_type,
                                num_run=num_run,
                                token_idx=token_idx,
                                timestamp=timestamp,
                                wall_time=wall_time,
                                itl_ms=itl_value * 1000,
                                token_text=str(text),
                            )
                            itl_callback(token_event)

                    most_recent_timestamp = timestamp
                    generated_text += text

            usage = getattr(event, "usage", None)
            if usage is not None:
                if hasattr(usage, "model_dump"):
                    usage_dict = usage.model_dump()
                else:
                    usage_dict = dict(usage)

                prompt_tokens = int(usage_dict.get("prompt_tokens", 0) or 0)
                completion_tokens = int(usage_dict.get("completion_tokens", 0) or 0)
                total_tokens = int(usage_dict.get("total_tokens", 0) or 0)

        latency = time.perf_counter() - start_time

        tpot = 0.0
        if completion_tokens > 1 and ttft > 0:
            tpot = (latency - ttft) / (completion_tokens - 1)

        if first_chunk_received:
            return RequestMetrics(
                user_id=user_id,
                user_type=user_type,
                num_run=num_run,
                prompt=prompt,
                answer=generated_text,
                latency=latency,
                status=200,
                ttft=ttft,
                tpot=tpot,
                itl=itl,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        return RequestMetrics(
            user_id=user_id,
            user_type=user_type,
            num_run=num_run,
            prompt=prompt,
            latency=latency,
            status=200,
            error="No valid response received",
            itl=itl,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


__all__ = ["LLMClient"]
