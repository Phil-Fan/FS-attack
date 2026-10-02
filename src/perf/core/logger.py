
from __future__ import annotations

import logging
import csv
import os
from datetime import datetime
from typing import List, Callable, Optional
from dataclasses import fields

from .results import RequestMetrics, TokenEvent


class CsvHandler(logging.Handler):

    def __init__(self, filename: str, fieldnames: List[str], mode: str = "a"):
        super().__init__()
        self.filename = filename
        self.fieldnames = fieldnames

        file_exists = os.path.exists(filename)
        self.file = open(filename, mode=mode, newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if not file_exists or mode == "w":
            self.writer.writeheader()

    def emit(self, record):
        row = {}
        asctime = self.formatter.formatTime(record)
        for key in self.fieldnames:
            if key == "msg" or key == "message":
                row[key] = record.getMessage()
            elif key == "timestamp":
                row[key] = asctime
            else:
                row[key] = getattr(record, key, "")
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()
        super().close()


class MetricsLogger:

    def __init__(
        self,
        result_logger: Optional[logging.Logger] = None,
        itl_logger: Optional[logging.Logger] = None,
    ):
        self._result_logger = result_logger
        self._itl_logger = itl_logger

    def log_result(self, metrics: RequestMetrics):
        if not self._result_logger:
            return

        user_type_display = "malicious" if metrics.user_type == "malicious" else "normal"
        if metrics.error:
            message = (
                f"[User {metrics.user_id:02d} ({user_type_display})] Failed | "
                f"Latency: {metrics.latency:.2f}s | Error: {metrics.error}"
            )
        else:
            ttft_str = f"TTFT: {metrics.ttft_ms:.2f}ms" if metrics.ttft > 0 else "TTFT: N/A"
            tpot_str = f"TPOT: {metrics.tpot_ms:.2f}ms" if metrics.tpot > 0 else "TPOT: N/A"
            message = (
                f"[User {metrics.user_id:02d} ({user_type_display})] Success | "
                f"Latency: {metrics.latency:.2f}s | Status: {metrics.status} | "
                f"{ttft_str} | {tpot_str}"
            )

        time_str = datetime.fromtimestamp(metrics.timestamp).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]

        self._result_logger.info(
            message,
            extra={
                "time": time_str,
                "user_id": metrics.user_id,
                "user_type": metrics.user_type,
                "num_run": metrics.num_run,
                "latency": f"{metrics.latency:.6f}",
                "ttft": f"{metrics.ttft_ms:.2f}" if metrics.ttft > 0 else "",
                "tpot": f"{metrics.tpot_ms:.2f}" if metrics.tpot > 0 else "",
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.completion_tokens,
                "total_tokens": metrics.total_tokens,
                "error": metrics.error or "",
                "tier": (metrics.tier or "") if metrics.user_type == "malicious" else "",
            },
        )

    def create_itl_callback(self) -> Optional[Callable[[TokenEvent], None]]:
        if not self._itl_logger:
            return None

        def callback(event: TokenEvent):
            self._itl_logger.info(
                "",
                extra={
                    "time": event.wall_time,
                    "user_id": event.user_id,
                    "user_type": event.user_type,
                    "num_run": event.num_run,
                    "token_idx": event.token_idx,
                    "itl_ms": f"{event.itl_ms:.4f}",
                    "token_text": event.token_text,
                    "kv_usage": event.kv_usage,
                },
            )

        return callback

    @classmethod
    def create_from_paths(
        cls,
        result_log_path: Optional[str] = None,
        itl_log_path: Optional[str] = None,
    ) -> "MetricsLogger":
        result_logger = None
        itl_logger = None

        if result_log_path:
            result_logger = get_logger(
                "result",
                result_log_path,
                is_csv=True,
                fieldnames=[
                    "time",
                    "user_id",
                    "user_type",
                    "num_run",
                    "latency",
                    "ttft",
                    "tpot",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "error",
                    "tier",
                ],
            )

        if itl_log_path:
            itl_logger = get_logger(
                "itl",
                itl_log_path,
                is_csv=True,
                fieldnames=[
                    "time",
                    "user_id",
                    "user_type",
                    "num_run",
                    "token_idx",
                    "itl_ms",
                    "token_text",
                    "kv_usage",
                ],
            )

        return cls(result_logger=result_logger, itl_logger=itl_logger)


def get_logger(
    name: str,
    output_path: str,
    is_csv: bool = False,
    fieldnames: Optional[List[str]] = None,
) -> logging.Logger:
    logger_name = f"{name}:{output_path}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger  # Avoid adding duplicate handlers

    logger.setLevel(logging.INFO)
    # Prevent bubbling to root handlers (avoid noisy stdout logs)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if is_csv:
        csv_handler = CsvHandler(output_path, fieldnames or [], mode="w")
        csv_handler.setFormatter(formatter)
        logger.addHandler(csv_handler)
    else:
        file_handler = logging.FileHandler(
            output_path, mode="w", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


__all__ = [
    "CsvHandler",
    "MetricsLogger",
    "get_logger",
    "setup_logging",
    "get_console_handler",
]


def get_console_handler(level: int = logging.INFO) -> logging.StreamHandler:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    return handler


def setup_logging(
    log_file: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    logger_names: Optional[List[str]] = None,
) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    root_logger.handlers.clear()

    console_handler = get_console_handler(console_level)
    root_logger.addHandler(console_handler)

    if log_file:
        import os
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    for logger_name in ["httpx", "httpcore", "openai", "urllib3"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    if logger_names:
        for name in logger_names:
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)
            logger.propagate = True
