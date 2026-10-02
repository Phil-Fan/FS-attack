#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import Optional

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

_CONFIGURED = False

_LEVEL_STYLES = {
    logging.DEBUG: ("DEBUG", "36"),
    logging.INFO: ("INFO", "32"),
    logging.WARNING: ("WARN", "33"),
    logging.ERROR: ("ERROR", "31"),
    logging.CRITICAL: ("CRIT", "41"),
}


def _supports_color(stream) -> bool:
    if stream is None:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(stream, "isatty") and stream.isatty()


class _ColorFormatter(logging.Formatter):
    def __init__(self, use_color: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelno
        level_name, color = _LEVEL_STYLES.get(level, (record.levelname, "37"))
        record.levelname = level_name
        msg = super().format(record)
        if not self._use_color:
            return msg
        return f"\x1b[{color}m{msg}\x1b[0m"


class _TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if tqdm is not None:
                tqdm.write(msg, file=self.stream)
            else:
                self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(level: Optional[str | int] = None, stream=None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")

    resolved_level = level
    if isinstance(level, str):
        resolved_level = logging._nameToLevel.get(level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(resolved_level)

    stream = stream or sys.stdout
    use_color = _supports_color(stream)

    handler = _TqdmLoggingHandler(stream=stream)
    formatter = _ColorFormatter(
        use_color=use_color,
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    handler.setLevel(resolved_level)

    # Avoid duplicate handlers
    for existing in root.handlers:
        if isinstance(existing, _TqdmLoggingHandler):
            _CONFIGURED = True
            return

    root.addHandler(handler)
    _CONFIGURED = True

    logging.captureWarnings(True)
    if os.environ.get("LOG_WARN_ONCE", "1") != "0":
        warnings.filterwarnings("once", category=UserWarning)
