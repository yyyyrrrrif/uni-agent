#!/usr/bin/env python3
"""Collect routing-decision rows from llm_router logs into a CSV table.

The KVCAware strategy emits two structured log prefixes (see
``kvc_aware.py`` / ``routing.py``):

    SCORE_ROW request_id=<rid> call_turns=<n> route_to=<replica>
        is_sticky=<bool> is_sticky_but_overload=<bool> is_combined=<bool>
        kv_usage=<f> running=<n> waiting=<n> load=<f> s_load=<f>
        s_cache=<f|-> gpu_hit=<f|->

    ROUTE_WINNER request_id=<rid> route_to=<replica>

Each ``SCORE_ROW`` is one table row. ``ROUTE_WINNER`` carries the
authoritative final pick under (possibly multi-strategy) weighting and
overrides the ``route_to_replica`` of the most recent same-``request_id``
``SCORE_ROW``.

Output columns (in this order):

    time, request_id, call_turns, route_to_replica, is_sticky,
    is_sticky_but_overload, is_combined, kv_usage, running, waiting,
    load, s_load, s_cache, gpu_hit

``time`` is the log line's timestamp (``YYYY-MM-DD HH:mm:ss`` + optional
``.fff``), extracted from the loguru/Ray prefix. Rows are emitted in log
order (i.e. chronological — loguru timestamps each line). Missing values
are kept as ``-``.

Usage:
    python examples/llm_router/collect_score_table.py router.log -o table.csv
    cat router.log | python examples/llm_router/collect_score_table.py -o table.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from typing import Iterable, TextIO

# Column order. ``time`` leads (log timestamp). The last two (s_cache, gpu_hit)
# are combined-path-only; sticky-hit rows keep them as "-".
COLUMNS: tuple[str, ...] = (
    "time",
    "request_id",
    "call_turns",
    "route_to_replica",
    "is_sticky",
    "is_sticky_but_overload",
    "is_combined",
    "kv_usage",
    "running",
    "waiting",
    "load",
    "s_load",
    "s_cache",
    "gpu_hit",
)

# Map SCORE_ROW keys → output column names (the few that differ).
_KEY_TO_COL: dict[str, str] = {
    "route_to": "route_to_replica",
}

# key="value" token. Values are bare words (no quoted strings in our format).
_TOKEN_RE = re.compile(r"(\w+)=(-|\S+)")

# Log line timestamp — loguru emits ``YYYY-MM-DD HH:mm:ss`` optionally
# followed by ``.fff`` millis. Ray's stdout prefix wraps the loguru line, so
# the timestamp sits after the ``(KVCAwareBalancer pid=…)`` tag; match the
# first occurrence anywhere on the line.
_TIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)")


@dataclass
class Row:
    """One SCORE_ROW, mutable so ROUTE_WINNER can overwrite route_to_replica."""

    fields: dict[str, str] = field(default_factory=dict)

    def get(self, col: str) -> str:
        return self.fields.get(col, "-")


def _parse_score_row(message: str) -> Row | None:
    """Parse a ``SCORE_ROW ...`` message into a Row, or None if not SCORE_ROW."""
    # The message is everything after the prefix; the log line may carry a
    # loguru timestamp/level prefix, so match SCORE_ROW anywhere on the line.
    idx = message.find("SCORE_ROW")
    if idx == -1:
        return None
    payload = message[idx + len("SCORE_ROW"):]
    fields: dict[str, str] = {}
    # Extract the log line's timestamp (search the whole line — Ray's prefix
    # puts it before SCORE_ROW, so payload alone wouldn't have it).
    time_match = _TIME_RE.search(message)
    if time_match is not None:
        fields["time"] = time_match.group(1)
    for key, val in _TOKEN_RE.findall(payload):
        fields[_KEY_TO_COL.get(key, key)] = val
    if "request_id" not in fields:
        return None  # malformed line — skip defensively
    return Row(fields=fields)


def _parse_route_winner(message: str) -> tuple[str, str] | None:
    """Parse a ``ROUTE_WINNER request_id=<rid> route_to=<replica>`` line.

    Returns ``(request_id, route_to)`` or None if not a ROUTE_WINNER line.
    """
    idx = message.find("ROUTE_WINNER")
    if idx == -1:
        return None
    payload = message[idx + len("ROUTE_WINNER"):]
    fields = dict(_TOKEN_RE.findall(payload))
    rid = fields.get("request_id")
    route_to = fields.get("route_to")
    if rid is None or route_to is None:
        return None
    return rid, route_to


def collect(lines: Iterable[str]) -> list[Row]:
    """Parse log lines into SCORE_ROW rows, back-filling ROUTE_WINNER picks.

    ROUTE_WINNER overrides ``route_to_replica`` on the most recent prior
    same-``request_id`` SCORE_ROW. If no such row exists yet (shouldn't
    happen in practice — route() always calls score() first), the winner is
    dropped.
    """
    rows: list[Row] = []
    last_row_by_request: dict[str, Row] = {}
    for line in lines:
        row = _parse_score_row(line)
        if row is not None:
            rows.append(row)
            last_row_by_request[row.get("request_id")] = row
            continue
        winner = _parse_route_winner(line)
        if winner is not None:
            rid, route_to = winner
            target = last_row_by_request.get(rid)
            if target is not None:
                target.fields["route_to_replica"] = route_to
    return rows


def write_csv(rows: list[Row], out: TextIO) -> None:
    writer = csv.writer(out)
    writer.writerow(COLUMNS)
    for row in rows:
        writer.writerow([row.get(col) for col in COLUMNS])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "logfile",
        nargs="?",
        default="-",
        help="Path to a router log file (default: stdin).",
    )
    parser.add_argument(
        "-o",
        "--out",
        dest="out",
        default="./route_time_analysis.csv",
        help="Path to write the CSV table. If omitted, write to stdout.",
    )
    args = parser.parse_args(argv)

    source: TextIO = sys.stdin if args.logfile == "-" else open(args.logfile, "r", encoding="utf-8")
    try:
        rows = collect(source)
    finally:
        if source is not sys.stdin:
            source.close()

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as out:
            write_csv(rows, out)
        print(f"wrote {len(rows)} rows → {args.out}", file=sys.stderr)
    else:
        write_csv(rows, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
