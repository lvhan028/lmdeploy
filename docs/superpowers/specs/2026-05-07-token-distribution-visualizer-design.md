# Token Distribution Visualizer

## Goal

A single Python script that scans eval-style JSONL message dumps, tokenizes each
conversation using a HuggingFace tokenizer with chat template, and produces
distribution visualizations (histogram, CDF, box plot) plus summary statistics,
rendered as PNGs embedded in an HTML report.

## Data Source

- **Location**: `/workspace/lmdeploy/workspace/z2_oc_infer_message_dump/cv21-fullbench2-dump2/`
- **342 JSONL files**, ~260K total rows
- **Format**: `{"message": [{"role": "user"|"system", "content": "..."}], "gold": [...]}`
- ~80% single-turn (1 message), ~20% two-turn (system + user)
- Categories auto-detected from filename prefix before first `-` or `_`

## Approach

Single-pass in-memory aggregation (Approach A). Load tokenizer, scan all files,
apply chat template + tokenize, accumulate counts per category, then generate
all plots and report. Data size (~260K integers) fits easily in memory.

## Script Location

`benchmark/visualize_token_distribution.py`

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset-dir` | `/workspace/lmdeploy/workspace/z2_oc_infer_message_dump/cv21-fullbench2-dump2` | Directory containing JSONL files |
| `--tokenizer` | (required) | HuggingFace tokenizer name, e.g. `Qwen/Qwen2.5-72B-Instruct` |
| `--output-dir` | `benchmark_outputs/token_dist_<timestamp>` | Output directory |
| `--max-files` | None | Limit number of JSONL files for quick runs |
| `--bins` | 50 | Number of histogram bins |

## Key Functions

- `parse_args()` — CLI argument parsing
- `detect_category(filename)` — extract prefix before first `-` or `_` in filename stem
- `count_tokens(messages, tokenizer)` — apply chat template + tokenize, return token count
- `collect_token_counts(dataset_dir, tokenizer, max_files)` — scan all JSONL, return `{category: [counts]}` and global list
- `compute_stats(token_counts)` — return dict with mean, median, p95, p99, min, max, count
- `_plot_histogram()`, `_plot_cdf()`, `_plot_box()`, `_plot_global_histogram()`, `_plot_global_cdf()` — matplotlib plotting
- `write_html_report()` — embed PNGs + stats table in HTML
- `main()` — orchestrate

## Token Counting Logic

1. Load tokenizer via `AutoTokenizer.from_pretrained(args.tokenizer)`
2. For each JSONL row: call `tokenizer.apply_chat_template(row["message"], tokenize=True)` and take `len()`
3. Skip rows with missing/empty messages
4. Group counts by category prefix

## Output Structure

```
benchmark_outputs/token_dist_YYYYMMDD_HHMMSS/
├── plots/
│   ├── global_histogram.png
│   ├── global_cdf.png
│   ├── category_histogram.png
│   ├── category_cdf.png
│   └── category_boxplot.png
├── stats.json
└── report.html
```

## HTML Report

- H1 title with dataset dir and tokenizer name
- Global histogram + CDF at top
- Per-category box plot
- Per-category histograms + CDFs (sequential)
- Summary stats table (one row per category + global row)
- Same CSS style as existing `benchmark_chat_completions.py` report

## Dependencies

- `transformers` (AutoTokenizer)
- `matplotlib` (Agg backend)
- Standard library: `json`, `math`, `argparse`, `pathlib`, `statistics`, `html`
