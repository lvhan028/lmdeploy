"""Visualize token distribution of messages in JSONL dumps.

Scans eval-style JSONL files, tokenizes each conversation using a HuggingFace
tokenizer with chat template, and produces distribution visualizations (histogram,
CDF, box plot) plus summary statistics, rendered as PNGs embedded in an HTML report.
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib
import json
import math
import statistics
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize token distribution of messages in JSONL dumps.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing eval JSONL files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[],
        help="Only process JSONL files whose stem contains one of these substrings (partial match).",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HuggingFace tokenizer name, e.g. Qwen/Qwen2.5-72B-Instruct.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to benchmark_outputs/token_dist_<timestamp>.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of JSONL files for quick runs.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path("benchmark_outputs") / f"token_dist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return args


def detect_category(filename: str) -> str:
    """Extract category prefix before the first '-' or '_' in a filename stem."""
    for ch in ("-", "_"):
        idx = filename.find(ch)
        if idx > 0:
            return filename[:idx]
    return filename


def count_tokens(messages: list[dict[str, Any]], tokenizer: Any) -> int:
    """Apply chat template and tokenize, return token count."""
    token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    token_ids = token_ids['input_ids']
    return len(token_ids)


def collect_token_counts(
    dataset_dir: Path,
    tokenizer: Any,
    max_files: int | None = None,
    datasets: Sequence[str] = (),
) -> tuple[dict[str, list[int]], list[int]]:
    """Scan all JSONL files and return per-category and global token counts."""
    files = sorted(dataset_dir.glob("*.jsonl"))
    if datasets:
        before = len(files)
        files = [f for f in files if any(d in f.stem for d in datasets)]
        print(f"Filter: {before} files -> {len(files)} files matching {datasets}")
    if max_files is not None:
        files = files[:max_files]

    category_counts: dict[str, list[int]] = {}
    global_counts: list[int] = []
    skipped = 0

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    iterator = tqdm(files, desc="Processing", unit="file") if tqdm else files
    for file_path in iterator:
        category = detect_category(file_path.stem)
        if category not in category_counts:
            category_counts[category] = []
        file_count = 0
        with file_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                messages = row.get("message")
                if not messages or not isinstance(messages, list):
                    skipped += 1
                    continue
                try:
                    n_tokens = count_tokens(messages, tokenizer)
                except Exception:
                    skipped += 1
                    continue
                category_counts[category].append(n_tokens)
                global_counts.append(n_tokens)
                file_count += 1
        if tqdm:
            iterator.set_postfix(file=file_path.name, msgs=file_count, refresh=False)
        else:
            print(f"  {file_path.name}: {file_count} messages")

    if skipped:
        print(f"Skipped {skipped} rows with missing/empty messages or tokenization errors.")
    print(f"Collected {len(global_counts)} token counts across {len(category_counts)} categories.")
    return category_counts, global_counts


def _percentile(sorted_values: list[int], q: float) -> float:
    """Compute the q-th percentile (0-100) of an already-sorted list."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    position = (n - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[int(position)])
    return float(sorted_values[lower] * (upper - position) + sorted_values[upper] * (position - lower))


def compute_stats(token_counts: list[int]) -> dict[str, Any]:
    """Return summary statistics for a list of token counts."""
    if not token_counts:
        return {
            "count": 0, "min": 0, "max": 0, "mean": 0.0,
            "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0,
        }
    ordered = sorted(token_counts)
    return {
        "count": len(token_counts),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(statistics.mean(token_counts), 2),
        "p50": round(_percentile(ordered, 50), 2),
        "p75": round(_percentile(ordered, 75), 2),
        "p90": round(_percentile(ordered, 90), 2),
        "p95": round(_percentile(ordered, 95), 2),
        "p99": round(_percentile(ordered, 99), 2),
    }


PERCENTILE_LINES = [50, 75, 90, 99]
PERCENTILE_COLORS = {
    50: "#2196F3",
    75: "#4CAF50",
    90: "#FF9800",
    99: "#F44336",
}


def _matplotlib_pyplot():
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        return importlib.import_module("matplotlib.pyplot")
    except Exception:
        return None


def _add_percentile_lines(ax, sorted_values: list[int], side: str = "right") -> None:
    """Draw vertical dashed lines for p50, p75, p90, p99 with a legend."""
    for p in PERCENTILE_LINES:
        val = _percentile(sorted_values, p)
        ax.axvline(val, color=PERCENTILE_COLORS[p], linestyle="--", linewidth=1.2,
                    label=f"p{p}={val:.0f}")
    if side == "right":
        ax.legend(fontsize="small", loc="upper right")
    else:
        ax.legend(fontsize="small", loc="upper left")


def _plot_global_histogram(
    output_dir: Path, global_counts: list[int], bins: int,
) -> Path | None:
    plt = _matplotlib_pyplot()
    if plt is None or not global_counts:
        return None

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(global_counts, bins=bins, alpha=0.75, color="#5C6BC0", edgecolor="white")
    sorted_global = sorted(global_counts)
    _add_percentile_lines(ax, sorted_global)
    ax.set_title("Global Token Distribution (Histogram)")
    ax.set_xlabel("Token count")
    ax.set_ylabel("Number of messages")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "global_histogram.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_global_cdf(
    output_dir: Path, global_counts: list[int],
) -> Path | None:
    plt = _matplotlib_pyplot()
    if plt is None or not global_counts:
        return None

    sorted_counts = sorted(global_counts)
    n = len(sorted_counts)
    cdf_y = [i / n for i in range(1, n + 1)]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(sorted_counts, cdf_y, color="#5C6BC0", linewidth=1.5)
    _add_percentile_lines(ax, sorted_counts, side="left")
    ax.set_title("Global Token Distribution (CDF)")
    ax.set_xlabel("Token count")
    ax.set_ylabel("Cumulative proportion")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "global_cdf.png"
    fig.savefig(path)
    plt.close(fig)
    return path


CATEGORY_COLORS = [
    "#5C6BC0", "#EF5350", "#66BB6A", "#FFA726", "#AB47BC",
    "#26C6DA", "#8D6E63", "#78909C", "#EC407A", "#7E57C2",
    "#29B6F6", "#9CCC65", "#FFCA28", "#FF7043", "#26A69A",
    "#42A5F5", "#D4E157", "#8D6E63", "#5C6BC0", "#78909C",
]


def _plot_category_histogram(
    output_dir: Path, category_counts: dict[str, list[int]],
    sorted_global: list[int], bins: int,
) -> Path | None:
    plt = _matplotlib_pyplot()
    if plt is None or not category_counts:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (cat, counts) in enumerate(sorted(category_counts.items())):
        color = CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
        ax.hist(counts, bins=bins, alpha=0.4, label=cat, color=color, edgecolor="white")
    _add_percentile_lines(ax, sorted_global)
    ax.set_title("Per-Category Token Distribution (Histogram)")
    ax.set_xlabel("Token count")
    ax.set_ylabel("Number of messages")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="x-small", ncol=3, loc="upper right")
    fig.tight_layout()
    path = output_dir / "category_histogram.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_category_cdf(
    output_dir: Path, category_counts: dict[str, list[int]],
    sorted_global: list[int],
) -> Path | None:
    plt = _matplotlib_pyplot()
    if plt is None or not category_counts:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (cat, counts) in enumerate(sorted(category_counts.items())):
        sorted_c = sorted(counts)
        n = len(sorted_c)
        color = CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
        ax.plot(sorted_c, [j / n for j in range(1, n + 1)], label=cat, color=color, linewidth=1.2)
    _add_percentile_lines(ax, sorted_global, side="left")
    ax.set_title("Per-Category Token Distribution (CDF)")
    ax.set_xlabel("Token count")
    ax.set_ylabel("Cumulative proportion")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="x-small", ncol=3, loc="upper left")
    fig.tight_layout()
    path = output_dir / "category_cdf.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_category_boxplot(
    output_dir: Path, category_counts: dict[str, list[int]],
) -> Path | None:
    plt = _matplotlib_pyplot()
    if plt is None or not category_counts:
        return None

    sorted_cats = sorted(category_counts.items(), key=lambda item: statistics.mean(item[1]), reverse=True)
    labels = [cat for cat, _ in sorted_cats]
    data = [counts for _, counts in sorted_cats]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.5), 6))
    ax.boxplot(data, tick_labels=labels, showfliers=False, vert=True)
    ax.set_title("Per-Category Token Distribution (Box Plot)")
    ax.set_ylabel("Token count")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize="x-small")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = output_dir / "category_boxplot.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_html_report(
    path: Path,
    all_stats: Sequence[dict[str, Any]],
    global_counts: list[int],
    plot_paths: Sequence[Path],
    dataset_dir: str,
    tokenizer_name: str,
    bins: int,
) -> None:
    """Write an HTML report embedding plot PNGs and summary stats table."""
    headers = ["category", "count", "min", "max", "mean", "p50", "p75", "p90", "p95", "p99"]
    header_html = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    table_rows = []
    for row in all_stats:
        cells = "".join(f"<td>{html.escape(str(row.get(h, '')))}</td>" for h in headers)
        table_rows.append(f"<tr>{cells}</tr>")

    plots_html = "\n".join(
        f'<figure><img src="{html.escape(str(p.relative_to(p.parent.parent)))}" alt="{html.escape(p.stem)}">'
        f"<figcaption>{html.escape(p.stem)}</figcaption></figure>"
        for p in plot_paths
    )

    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Token Distribution Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 0.35rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ max-width: 100%; }}
    .chart {{ width: 100%; height: 440px; margin-bottom: 2rem; }}
  </style>
</head>
<body>
  <h1>Token Distribution Report</h1>
  <p>Dataset: {html.escape(str(dataset_dir))}</p>
  <p>Tokenizer: {html.escape(tokenizer_name)}</p>
  <p>Histogram bins: {html.escape(str(bins))}. Total messages: {html.escape(str(len(global_counts)))}.</p>
  <h2>Plots</h2>
  {plots_html}
  <h2>Summary Statistics</h2>
  <table><thead><tr>{header_html}</tr></thead><tbody>{"".join(table_rows)}</tbody></table>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Scanning JSONL files in: {args.dataset_dir}")
    category_counts, global_counts = collect_token_counts(
        args.dataset_dir, tokenizer, args.max_files, args.datasets,
    )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    sorted_global = sorted(global_counts)

    print("Generating plots...")
    plot_steps = [
        ("Global histogram", lambda: _plot_global_histogram(plots_dir, global_counts, args.bins)),
        ("Global CDF", lambda: _plot_global_cdf(plots_dir, global_counts)),
        ("Category histogram", lambda: _plot_category_histogram(plots_dir, category_counts, sorted_global, args.bins)),
        ("Category CDF", lambda: _plot_category_cdf(plots_dir, category_counts, sorted_global)),
        ("Category box plot", lambda: _plot_category_boxplot(plots_dir, category_counts)),
    ]
    plot_paths = []
    for idx, (label, fn) in enumerate(plot_steps, 1):
        print(f"  [{idx}/{len(plot_steps)}] {label}...")
        p = fn()
        if p is not None:
            plot_paths.append(p)

    print("Writing stats and report...")
    all_stats: list[dict[str, Any]] = []
    for cat in sorted(category_counts):
        stats = compute_stats(category_counts[cat])
        stats["category"] = cat
        all_stats.append(stats)
    global_stats = compute_stats(global_counts)
    global_stats["category"] = "GLOBAL"
    all_stats.append(global_stats)

    (output_dir / "stats.json").write_text(
        json.dumps(all_stats, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    all_stats.sort(key=lambda row: row["p99"])

    csv_path = output_dir / "stats.csv"
    csv_headers = ["category", "count", "min", "max", "mean", "p50", "p75", "p90", "p95", "p99"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(all_stats)

    write_html_report(
        output_dir / "report.html",
        all_stats,
        global_counts,
        plot_paths,
        str(args.dataset_dir),
        args.tokenizer,
        args.bins,
    )

    print(f"Report written to: {output_dir / 'report.html'}")
    print(f"Stats written to: {output_dir / 'stats.json'}")
    print(f"CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
