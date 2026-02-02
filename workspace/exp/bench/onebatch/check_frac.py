#!/usr/bin/env python3
"""
Summarize experiment outcomes by mem fraction (x-axis) and batch size (y-axis).

For each (method, model) pair, generate a separate table where each cell reports:
  ok / oom / limit counts

This script reuses the filename/content parsing logic from `log_parser.py`.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# Make sibling import work no matter where the script is executed from.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from log_parser import Info, get_infos_from_dir  # noqa: E402


def _safe_name(s: str) -> str:
    # Keep filenames readable; avoid path separators and spaces.
    return (
        s.replace(os.sep, "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("__", "_")
    )


def _fmt_frac(x: float) -> str:
    # Stable pretty formatting for columns.
    # Use up to 4 decimals, but trim trailing zeros/dot.
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _round_frac_key(x: float) -> float:
    # Group floats robustly (fractions typically have <=2-3 decimals).
    return round(float(x), 4)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _to_markdown_table(
    row_labels: List[str],
    col_labels: List[str],
    cells: List[List[str]],
) -> str:
    # Simple GitHub-flavored markdown table.
    header = ["bsz\\frac"] + col_labels
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r, row in zip(row_labels, cells):
        lines.append("| " + " | ".join([r] + row) + " |")
    return "\n".join(lines) + "\n"


def _to_tsv(
    row_labels: List[str],
    col_labels: List[str],
    cells: List[List[str]],
) -> str:
    header = ["bsz\\frac"] + col_labels
    lines = ["\t".join(header)]
    for r, row in zip(row_labels, cells):
        lines.append("\t".join([r] + row))
    return "\n".join(lines) + "\n"


def summarize_directory(input_dir: str) -> Tuple[Dict[Tuple[str, str], List[Info]], Dict[str, int]]:
    infos = get_infos_from_dir(input_dir)
    grouped: Dict[Tuple[str, str], List[Info]] = defaultdict(list)
    status_counts: Dict[str, int] = defaultdict(int)

    for info in infos:
        grouped[(info.method, info.model)].append(info)
        status_counts[info.status] += 1

    return grouped, dict(status_counts)


def build_frac_bsz_table(infos: Iterable[Info]) -> Tuple[List[str], List[str], List[List[str]], Dict[str, int]]:
    # rows: bsz, cols: mem_frac
    # cell: "ok/oom/limit"
    by_cell: Dict[Tuple[int, float], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    status_counts: Dict[str, int] = defaultdict(int)

    bsz_set = set()
    frac_set = set()

    for info in infos:
        bsz_set.add(info.bsz)
        frac_key = _round_frac_key(info.mem_frac)
        frac_set.add(frac_key)
        by_cell[(info.bsz, frac_key)][info.status] += 1
        status_counts[info.status] += 1

    bszs = sorted(bsz_set)
    fracs = sorted(frac_set)

    row_labels = [str(b) for b in bszs]
    col_labels = [_fmt_frac(f) for f in fracs]

    cells: List[List[str]] = []
    for b in bszs:
        row: List[str] = []
        for f in fracs:
            c = by_cell.get((b, f), {})
            ok = int(c.get("ok", 0))
            oom = int(c.get("oom", 0))
            limit = int(c.get("limit", 0))
            row.append(f"{ok}/{oom}/{limit}")
        cells.append(row)

    return row_labels, col_labels, cells, dict(status_counts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize mem_frac × batchsize success/oom/limit counts.")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing experiment logs (recursively searched).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="check_frac",
        help="Output directory for generated tables. Default: check_frac",
    )
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Include 'error' status in per-cell output as a fourth number (ok/oom/limit/error). Default off.",
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped, status_counts = summarize_directory(input_dir)

    # Top-level summary
    summary_lines = [
        f"input_dir: {Path(input_dir).resolve()}",
        f"total_logs_parsed: {sum(status_counts.values())}",
        "status_counts:",
    ]
    for k in sorted(status_counts.keys()):
        summary_lines.append(f"  {k}: {status_counts[k]}")
    summary_lines.append("")

    _write_text(out_dir / "summary.txt", "\n".join(summary_lines) + "\n")

    if not grouped:
        _write_text(out_dir / "README.md", "No parsable log files found.\n")
        print(f"[check_frac] No parsable log files found under: {input_dir}")
        print(f"[check_frac] Wrote: {out_dir}/summary.txt")
        return 0

    # Generate one table per (method, model)
    index_lines = [
        "# check_frac outputs",
        "",
        f"- **input_dir**: `{Path(input_dir).resolve()}`",
        f"- **total_logs_parsed**: `{sum(status_counts.values())}`",
        "",
        "## Tables (one per method/model)",
        "",
        "Each cell is `ok/oom/limit` counts aggregated across all other params (in/out/pg/ctx/etc) under the same `(bsz, mem_frac)`.",
        "",
    ]

    for (method, model), infos in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        row_labels, col_labels, cells, group_status_counts = build_frac_bsz_table(infos)

        # Optionally include error as 4th number
        if args.include_errors:
            # rebuild cells to include error
            by_cell: Dict[Tuple[int, float], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
            bszs = sorted({i.bsz for i in infos})
            fracs = sorted({_round_frac_key(i.mem_frac) for i in infos})
            for i in infos:
                by_cell[(i.bsz, _round_frac_key(i.mem_frac))][i.status] += 1
            row_labels = [str(b) for b in bszs]
            col_labels = [_fmt_frac(f) for f in fracs]
            cells = []
            for b in bszs:
                row = []
                for f in fracs:
                    c = by_cell.get((b, f), {})
                    ok = int(c.get("ok", 0))
                    oom = int(c.get("oom", 0))
                    limit = int(c.get("limit", 0))
                    err = int(c.get("error", 0))
                    row.append(f"{ok}/{oom}/{limit}/{err}")
                cells.append(row)

        name = f"method_{_safe_name(method)}__model_{_safe_name(model)}"
        md_path = out_dir / f"{name}.md"
        tsv_path = out_dir / f"{name}.tsv"

        md = []
        md.append(f"# {name}")
        md.append("")
        md.append(f"- **method**: `{method}`")
        md.append(f"- **model**: `{model}`")
        md.append(f"- **num_logs**: `{len(infos)}`")
        md.append("- **status_counts**:")
        for k in sorted(group_status_counts.keys()):
            md.append(f"  - `{k}`: `{group_status_counts[k]}`")
        md.append("")
        md.append("Cell format: `ok/oom/limit`" + (" (or `ok/oom/limit/error`)" if args.include_errors else ""))
        md.append("")
        md.append(_to_markdown_table(row_labels, col_labels, cells))

        _write_text(md_path, "\n".join(md))
        _write_text(tsv_path, _to_tsv(row_labels, col_labels, cells))

        index_lines.append(f"- `{md_path.name}` (and `{tsv_path.name}`)")

    index_lines.append("")
    _write_text(out_dir / "README.md", "\n".join(index_lines) + "\n")

    print(f"[check_frac] Parsed logs under: {input_dir}")
    print(f"[check_frac] Wrote outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

