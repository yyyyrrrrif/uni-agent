"""Centralised logging for the llm_router package.

All llm_router components run inside a Ray actor process where no root
logger handler is pre-configured — INFO-level messages would be swallowed.
This module ensures loguru has a stdout sink so routing decisions reach
Ray's captured log stream, and provides ``get_router_logger()`` for
per-component bound loggers.

The project standard is loguru (``uni_agent.async_logging``). This module
replaces the copy-pasted ``logging.StreamHandler`` blocks that were
duplicated across 4 llm_router files.

Optional file sink: set the ``LLM_ROUTER_LOG_FILE`` env var to a file path
to mirror every router log line to that file. Ray multiplexes each actor's
stdout into one captured stream, which interleaves and truncates lines
under load (observed: SCORE_ROW lines lost between actors). The file sink
bypasses Ray's stdout mux — each actor process appends directly — and
``enqueue=True`` serialises writes within a process, so structured
SCORE_ROW / ROUTE_WINNER lines land whole. Default: off (no env var set).
"""

from __future__ import annotations

import os
import sys

import loguru
from loguru import logger

# ── Ensure a stdout sink exists for the Ray actor process ────────────────
# Equivalent to the old per-file ``if not logger.handlers: … StreamHandler
# + propagate=False`` block, but done once centrally.
_stream_sink_id: int | None = None
_file_sink_id: int | None = None

# Format shared by stdout and file sinks — the SCORE_ROW / ROUTE_WINNER
# collection script (examples/llm_router/collect_score_table.py) parses
# the message portion, which this format keeps after the level column.
_ROUTER_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {extra[name]: <20} | {level: <8} | {message}"


def _ensure_stdout_sink() -> None:
    """Add a loguru stdout sink if none exists yet.

    Called at module import so that any ``get_router_logger()`` call
    immediately produces visible output, even inside a bare Ray actor
    process where loguru's default handler was removed by
    ``async_logging.py``.
    """
    global _stream_sink_id
    if _stream_sink_id is not None:
        return
    # Check whether any StreamSink (stdout/stderr) handler already exists
    # from async_logging's DEBUG_MODE setup — don't add a duplicate.
    for h in logger._core.handlers.values():
        sink = h._sink
        if hasattr(sink, "_stream") and sink._stream in (sys.stdout, sys.stderr):
            return
    _stream_sink_id = logger.add(
        sys.stdout,
        level="INFO",
        format=_ROUTER_FORMAT,
    )


def _ensure_file_sink() -> None:
    """Add a loguru file sink when ``LLM_ROUTER_LOG_FILE`` is set.

    Bypasses Ray's interleaved stdout mux: each actor process appends
    directly to the file, and ``enqueue=True`` serialises writes within the
    process so a multi-line ``logger.info`` sequence (the score() per-replica
    rows + SCORE_ROW) is not interleaved with another thread's. Append mode
    lets multiple actor processes share one file without clobbering each
    other's output. Default: off — the env var gates this so production runs
    that don't need file logging pay no overhead.

    Idempotent and re-entrant: if ``uni_agent.async_logging`` ran
    ``logger.remove()`` after this module's import (it does, at its own
    import time), the sink added here is gone. Callers that run later in the
    process lifecycle — notably ``KVCAwareBalancer.__init__``, which runs
    after all imports are settled — re-invoke this to restore the sink.
    """
    global _file_sink_id
    path = os.environ.get("LLM_ROUTER_LOG_FILE")
    if not path:
        return
    # If our sink is still live, nothing to do. (loguru assigns each add()
    # a small int id; a live id means the sink survived any remove() calls.)
    if _file_sink_id is not None and _file_sink_id in logger._core.handlers:
        return
    level = os.environ.get("LLM_ROUTER_LOG_LEVEL", "INFO").upper()
    _file_sink_id = logger.add(
        path,
        level=level,
        format=_ROUTER_FORMAT,
        mode="a",
        enqueue=True,
    )


_ensure_stdout_sink()
_ensure_file_sink()


# ── Per-component logger factory ─────────────────────────────────────────


def get_router_logger(name: str) -> loguru.Logger:
    """Return a loguru bound logger for an llm_router component.

    Unlike the full ``async_logging.get_logger(name, run_id)`` which binds
    a ``run_id``, llm_router components operate inside a Ray actor without
    per-run context. We bind only ``name`` for readable log output.

    Args:
        name: Human-readable component label (e.g. ``"balancer"``).
              Appears in the ``{extra[name]}`` column of the format string.

    Returns:
        A loguru ``BoundLogger`` that can be used as ``logger.info(…)`` etc.
    """
    return logger.bind(name=name)
