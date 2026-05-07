"""Benchmark OpenAI-compatible /v1/chat/completions endpoints.

This script focuses on eval-style JSONL dumps where each row contains OpenAI
chat ``messages``. It records streaming latency traces, aggregates TTFT/ITL/TPOT
metrics, and writes table plus report artifacts for concurrency/RPS sweeps.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import html
import importlib
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkRequest:
    dataset: str
    id: str
    messages: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SSEEvent:
    content: str = ''
    reasoning_content: str = ''
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    done: bool = False
    raw: dict[str, Any] | None = None

    @property
    def token_text(self) -> str:
        return self.content or self.reasoning_content


@dataclass
class RequestTrace:
    dataset: str
    request_id: str
    mode: str
    setting: float
    repeat: int
    success: bool
    start_time: float = 0.0
    first_token_time: float | None = None
    end_time: float = 0.0
    chunk_times: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage_available: bool = False
    generated_text: str = ''
    reasoning_text: str = ''
    finish_reason: str | None = None
    http_status: int | None = None
    error: str = ''

    @property
    def ttft_s(self) -> float:
        if self.first_token_time is None:
            return 0.0
        return max(self.first_token_time - self.start_time, 0.0)

    @property
    def e2e_latency_s(self) -> float:
        return max(self.end_time - self.start_time, 0.0)

    @property
    def itls_s(self) -> list[float]:
        return [
            max(self.chunk_times[idx] - self.chunk_times[idx - 1], 0.0)
            for idx in range(1, len(self.chunk_times))
        ]

    @property
    def tpot_s(self) -> float:
        if self.first_token_time is None or self.completion_tokens <= 0:
            return 0.0
        denominator = max(self.completion_tokens - 1, 1)
        return max(self.end_time - self.first_token_time, 0.0) / denominator


SendOne = Callable[[BenchmarkRequest, str, float, int], Awaitable[RequestTrace]]


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(',') if item.strip()]


def _parse_number_list(value: str, as_int: bool = False) -> list[int | float]:
    numbers: list[int | float] = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        numbers.append(int(item) if as_int else float(item))
    if not numbers:
        raise ValueError('list argument must contain at least one value')
    return numbers


def _discover_dataset_files(dataset_dir: Path | None, dataset_files: Sequence[Path] | None) -> list[Path]:
    if dataset_files:
        return sorted(Path(path) for path in dataset_files)
    if dataset_dir is None:
        raise ValueError('Either dataset_dir or dataset_files must be provided.')
    return sorted(Path(dataset_dir).glob('*.jsonl'))


def _dataset_matches(dataset: str, selected: Sequence[str]) -> bool:
    return any(dataset == item or dataset.startswith(item) for item in selected)


def _normalize_row(row: dict[str, Any], dataset: str, row_index: int) -> BenchmarkRequest:
    messages = row.get('messages') or row.get('message')
    if messages is None:
        prompt = row.get('prompt')
        if prompt is None:
            raise ValueError(f"row {row_index} in {dataset} must contain either messages or prompt")
        messages = [{'role': 'user', 'content': prompt}]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"row {row_index} in {dataset} has invalid messages")
    request_id = str(row.get('id', f"{dataset}-{row_index}"))
    metadata = row.get('metadata', {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {'value': metadata}
    return BenchmarkRequest(dataset=dataset, id=request_id, messages=messages, metadata=metadata)


def load_requests(
    dataset_dir: str | Path | None = None,
    dataset_files: Sequence[str | Path] | None = None,
    datasets: Sequence[str] | None = None,
    num_prompts: int | None = None,
    shuffle: bool = False,
    seed: int = 1,
) -> list[BenchmarkRequest]:
    """Load JSONL chat requests.

    ``num_prompts`` is applied per dataset file so datasets with different sizes
    remain balanced in sweeps.
    """
    selected = list(dict.fromkeys(datasets or []))
    files = _discover_dataset_files(
        Path(dataset_dir) if dataset_dir is not None else None,
        [Path(path) for path in dataset_files] if dataset_files is not None else None,
    )

    all_requests: list[BenchmarkRequest] = []
    for file_path in files:
        dataset = file_path.stem
        if selected and not _dataset_matches(dataset, selected):
            continue
        rows: list[BenchmarkRequest] = []
        with file_path.open(encoding='utf-8') as f:
            for row_index, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rows.append(_normalize_row(json.loads(line), dataset, row_index))
        if shuffle:
            random.Random(seed).shuffle(rows)
        if num_prompts is not None:
            rows = rows[:num_prompts]
        all_requests.extend(rows)

    if not all_requests:
        raise ValueError('No benchmark requests were loaded.')
    return all_requests


def parse_sse_line(line: bytes | str) -> SSEEvent:
    if isinstance(line, bytes):
        line = line.decode('utf-8')
    line = line.strip()
    if not line:
        return SSEEvent()
    if line.startswith('data:'):
        line = line[len('data:'):].strip()
    if line == '[DONE]':
        return SSEEvent(done=True)

    data = json.loads(line)
    choice = (data.get('choices') or [{}])[0]
    delta = choice.get('delta') or {}
    return SSEEvent(
        content=delta.get('content') or '',
        reasoning_content=delta.get('reasoning_content') or '',
        finish_reason=choice.get('finish_reason'),
        usage=data.get('usage'),
        raw=data,
    )


def build_payload(
    request: BenchmarkRequest,
    model: str,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_completion_tokens: int | None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'model': model,
        'messages': request.messages,
        'temperature': temperature,
        'top_p': top_p,
        'stream': True,
        'stream_options': {'include_usage': True},
    }
    if top_k is not None:
        payload['top_k'] = top_k
    if max_completion_tokens is not None:
        payload['max_completion_tokens'] = max_completion_tokens
    if extra_body:
        payload.update(extra_body)
    return payload


def _chat_completions_url(base_url: str, api_path: str) -> str:
    base_url = base_url.rstrip('/')
    api_path = api_path.strip()
    if not api_path:
        api_path = '/chat/completions' if base_url.endswith('/v1') else '/v1/chat/completions'
    if not api_path.startswith('/'):
        api_path = f"/{api_path}"
    return f"{base_url}{api_path}"


def _models_url(base_url: str) -> str:
    base_url = base_url.rstrip('/')
    if base_url.endswith('/v1'):
        return f"{base_url}/models"
    return f"{base_url}/v1/models"


async def fetch_model_id(session: Any, base_url: str, api_key: str | None = None) -> str:
    headers = {'Authorization': f"Bearer {api_key}"} if api_key else None
    async with session.get(_models_url(base_url), headers=headers) as response:
        if response.status != 200:
            raise RuntimeError(f"Failed to fetch model from /v1/models: {response.status} {response.reason}")
        payload = await response.json()
    model_list = payload.get('data') or []
    if not model_list or not model_list[0].get('id'):
        raise RuntimeError('No model id found in /v1/models response.')
    return str(model_list[0]['id'])


async def request_chat_completion(
    session: Any,
    request: BenchmarkRequest,
    url: str,
    model: str,
    mode: str,
    setting: float,
    repeat: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_completion_tokens: int | None,
    extra_body: dict[str, Any] | None,
    headers: dict[str, str] | None = None,
    save_response_text: bool = False,
) -> RequestTrace:
    payload = build_payload(
        request=request,
        model=model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_completion_tokens=max_completion_tokens,
        extra_body=extra_body,
    )
    trace = RequestTrace(
        dataset=request.dataset,
        request_id=request.id,
        mode=mode,
        setting=setting,
        repeat=repeat,
        success=False,
        start_time=time.perf_counter(),
    )
    try:
        async with session.post(url, json=payload, headers=headers) as response:
            trace.http_status = response.status
            if response.status != 200:
                trace.error = f"{response.status} {response.reason}: {await response.text()}"
                trace.end_time = time.perf_counter()
                return trace

            async for chunk in response.content:
                for raw_line in chunk.splitlines():
                    if not raw_line.strip():
                        continue
                    event = parse_sse_line(raw_line)
                    if event.done:
                        continue
                    now = time.perf_counter()
                    if event.token_text:
                        if trace.first_token_time is None:
                            trace.first_token_time = now
                        trace.chunk_times.append(now)
                    if save_response_text and event.content:
                        trace.generated_text += event.content
                    if save_response_text and event.reasoning_content:
                        trace.reasoning_text += event.reasoning_content
                    if event.finish_reason is not None:
                        trace.finish_reason = event.finish_reason
                    if event.usage:
                        trace.usage_available = True
                        trace.prompt_tokens = int(event.usage.get('prompt_tokens', trace.prompt_tokens) or 0)
                        trace.completion_tokens = int(
                            event.usage.get('completion_tokens', trace.completion_tokens) or 0
                        )
            trace.end_time = time.perf_counter()
            trace.success = trace.error == ''
            return trace
    except Exception as e:  # noqa: BLE001 - benchmark should record failures and continue.
        trace.end_time = time.perf_counter()
        trace.error = repr(e)
        return trace


async def closed_loop_runner(
    requests: Sequence[BenchmarkRequest],
    concurrency: int,
    repeat: int,
    send_one: SendOne,
) -> list[RequestTrace]:
    queue: asyncio.Queue[BenchmarkRequest] = asyncio.Queue()
    for request in requests:
        queue.put_nowait(request)

    traces: list[RequestTrace] = []

    async def worker():
        while True:
            try:
                request = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                traces.append(await send_one(request, 'concurrency', float(concurrency), repeat))
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(max(concurrency, 1))]
    await asyncio.gather(*workers)
    return traces


async def request_rate_runner(
    requests: Sequence[BenchmarkRequest],
    request_rate: float,
    repeat: int,
    send_one: SendOne,
    seed: int = 1,
) -> list[RequestTrace]:
    rng = random.Random(seed)
    tasks: list[asyncio.Task[RequestTrace]] = []
    for request in requests:
        tasks.append(asyncio.create_task(send_one(request, 'request-rate', float(request_rate), repeat)))
        if request_rate != float('inf'):
            await asyncio.sleep(rng.expovariate(request_rate))
    return list(await asyncio.gather(*tasks))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    return float(ordered[lower] * (upper - position) + ordered[upper] * (position - lower))


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _latency_stats(prefix: str, values_s: Sequence[float]) -> dict[str, float]:
    values_ms = [value * 1000 for value in values_s]
    if not values_ms:
        return {
            f"ave_{prefix}_ms": 0.0,
            f"min_{prefix}_ms": 0.0,
            f"max_{prefix}_ms": 0.0,
            f"p50_{prefix}_ms": 0.0,
            f"p75_{prefix}_ms": 0.0,
            f"p95_{prefix}_ms": 0.0,
            f"p99_{prefix}_ms": 0.0,
        }
    return {
        f"ave_{prefix}_ms": _mean(values_ms),
        f"min_{prefix}_ms": min(values_ms),
        f"max_{prefix}_ms": max(values_ms),
        f"p50_{prefix}_ms": percentile(values_ms, 50),
        f"p75_{prefix}_ms": percentile(values_ms, 75),
        f"p95_{prefix}_ms": percentile(values_ms, 95),
        f"p99_{prefix}_ms": percentile(values_ms, 99),
    }


def token_distribution_stats(values: Sequence[int]) -> list[tuple[str, float]]:
    if not values:
        return []
    float_values = [float(value) for value in values]
    return [
        ('mean', _mean(float_values)),
        ('p50', percentile(float_values, 50)),
        ('p75', percentile(float_values, 75)),
        ('p90', percentile(float_values, 90)),
        ('p99', percentile(float_values, 99)),
    ]


TOKEN_STAT_COLORS = {
    'mean': 'tab:red',
    'p50': 'tab:orange',
    'p75': 'tab:green',
    'p90': 'tab:blue',
    'p99': 'tab:purple',
}


def _group_key(trace: RequestTrace) -> tuple[str, str, float, int]:
    return (trace.dataset, trace.mode, trace.setting, trace.repeat)


def aggregate_traces(traces: Sequence[RequestTrace]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, float, int], list[RequestTrace]] = defaultdict(list)
    for trace in traces:
        grouped[_group_key(trace)].append(trace)

    for (dataset, mode, setting, repeat), group in sorted(grouped.items()):
        completed = [trace for trace in group if trace.success]
        failed = len(group) - len(completed)
        start = min((trace.start_time for trace in group), default=0.0)
        end = max((trace.end_time for trace in group), default=start)
        duration = max(end - start, 0.0)
        total_input = sum(trace.prompt_tokens for trace in completed)
        total_output = sum(trace.completion_tokens for trace in completed)
        itls = [itl for trace in completed for itl in trace.itls_s]

        summary: dict[str, Any] = {
            'dataset': dataset,
            'mode': mode,
            'setting': setting,
            'repeat': repeat,
            'total': len(group),
            'completed': len(completed),
            'failed': failed,
            'success_rate': len(completed) / len(group) if group else 0.0,
            'duration_s': duration,
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'request_throughput_req_s': len(completed) / duration if duration > 0 else 0.0,
            'input_throughput_tok_s': total_input / duration if duration > 0 else 0.0,
            'output_throughput_tok_s': total_output / duration if duration > 0 else 0.0,
        }
        summary.update(_latency_stats('ttft', [trace.ttft_s for trace in completed if trace.first_token_time]))
        summary.update(_latency_stats('itl', itls))
        summary['usage_available'] = all(trace.usage_available for trace in completed) if completed else False
        summary.update(_latency_stats(
            'tpot',
            [trace.tpot_s for trace in completed if trace.first_token_time and trace.completion_tokens > 0],
        ))
        summary.update(_latency_stats('e2e_latency', [trace.e2e_latency_s for trace in completed]))
        summaries.append(summary)
    return summaries


def _trace_to_json(trace: RequestTrace) -> dict[str, Any]:
    item = asdict(trace)
    item['ttft_s'] = trace.ttft_s
    item['itls_s'] = trace.itls_s
    item['tpot_s'] = trace.tpot_s
    item['e2e_latency_s'] = trace.e2e_latency_s
    return item


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _write_requests_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        'dataset',
        'request_id',
        'mode',
        'setting',
        'repeat',
        'success',
        'http_status',
        'ttft_s',
        'tpot_s',
        'e2e_latency_s',
        'prompt_tokens',
        'completion_tokens',
        'usage_available',
        'finish_reason',
        'error',
    ]
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(path: Path, summaries: Sequence[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for summary in summaries:
        for key in summary:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def _matplotlib_pyplot():
    try:
        matplotlib = importlib.import_module('matplotlib')
        matplotlib.use('Agg')
        return importlib.import_module('matplotlib.pyplot')
    except Exception:
        return None


def _mode_axis_label(mode: str) -> str:
    if mode == 'concurrency':
        return 'Concurrency'
    if mode == 'request-rate':
        return 'Request rate (req/s)'
    return 'Benchmark setting'


def _plot_metric(
    output_dir: Path,
    summaries: Sequence[dict[str, Any]],
    metric: str,
    title: str,
    mode: str,
) -> Path | None:
    plt = _matplotlib_pyplot()
    if plt is None:
        return None

    by_series: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        if metric in summary:
            by_series[(str(summary['dataset']), str(summary['mode']))].append(summary)

    if not by_series:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    for (dataset, mode), rows in sorted(by_series.items()):
        rows = sorted(rows, key=lambda item: float(item['setting']))
        ax.plot(
            [float(item['setting']) for item in rows],
            [float(item.get(metric, 0.0)) for item in rows],
            marker='o',
            label=f"{dataset} ({mode})",
        )
    ax.set_title(title)
    ax.set_xlabel(_mode_axis_label(mode))
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='small')
    fig.tight_layout()
    path = output_dir / f"{metric}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_latency_stats(
    output_dir: Path,
    summaries: Sequence[dict[str, Any]],
    prefix: str,
    title: str,
    mode: str,
) -> Path | None:
    plt = _matplotlib_pyplot()
    if plt is None:
        return None

    stats = [
        (f"ave_{prefix}_ms", 'ave'),
        (f"p50_{prefix}_ms", 'p50'),
        (f"p75_{prefix}_ms", 'p75'),
        (f"p95_{prefix}_ms", 'p95'),
        (f"p99_{prefix}_ms", 'p99'),
    ]
    by_series: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        if any(key in summary for key, _ in stats):
            by_series[(str(summary['dataset']), str(summary['mode']))].append(summary)

    if not by_series:
        return None

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (dataset, mode), rows in sorted(by_series.items()):
        rows = sorted(rows, key=lambda item: float(item['setting']))
        x_values = [float(item['setting']) for item in rows]
        for key, stat_label in stats:
            ax.plot(
                x_values,
                [float(item.get(key, 0.0)) for item in rows],
                marker='o',
                label=f"{dataset} ({mode}) {stat_label}",
            )
    ax.set_title(title)
    ax.set_xlabel(_mode_axis_label(mode))
    ax.set_ylabel('Latency (ms)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='x-small', ncol=2)
    fig.tight_layout()
    path = output_dir / f"{prefix}_latency_stats.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_token_histogram(output_dir: Path,
                          traces: Sequence[RequestTrace],
                          token_field: str, title: str) -> Path | None:
    plt = _matplotlib_pyplot()
    if plt is None:
        return None

    values = [int(getattr(trace, token_field)) for trace in traces if trace.success and getattr(trace, token_field) > 0]
    if not values:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = min(max(int(math.sqrt(len(values))), 10), 80)
    ax.hist(values, bins=bins, alpha=0.85)
    for label, value in token_distribution_stats(values):
        ax.axvline(
            value,
            color=TOKEN_STAT_COLORS[label],
            linestyle='--',
            linewidth=1.4,
            label=f"{label}: {value:.1f}",
        )
    ax.set_title(title)
    ax.set_xlabel('Tokens')
    ax.set_ylabel('Request count')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='small')
    fig.tight_layout()
    path = output_dir / f"{token_field}_histogram.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _write_html_report(path: Path, summaries: Sequence[dict[str, Any]], plot_paths: Sequence[Path], mode: str) -> None:
    headers = list(summaries[0].keys()) if summaries else []
    table_rows = []
    for summary in summaries:
        cells = ''.join(f"<td>{html.escape(str(summary.get(header, '')))}</td>" for header in headers)
        table_rows.append(f"<tr>{cells}</tr>")
    header_html = ''.join(f"<th>{html.escape(header)}</th>" for header in headers)
    plots_html = '\n'.join(
        f'<figure><img src="{html.escape(str(path.relative_to(path.parent.parent)))}" alt="{html.escape(path.stem)}">'
        f"<figcaption>{html.escape(path.stem)}</figcaption></figure>"
        for path in plot_paths
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Chat Completions Benchmark Report</title>
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
  <h1>Chat Completions Benchmark Report</h1>
  <p>Mode: {html.escape(mode)}. X-axis: {html.escape(_mode_axis_label(mode))}.</p>
  <p>PNG plots are generated from benchmark summary and request token statistics.</p>
  <h2>Plots</h2>
  {plots_html}
  <h2>Summary</h2>
  <table><thead><tr>{header_html}</tr></thead><tbody>{''.join(table_rows)}</tbody></table>
</body>
</html>
""",
        encoding='utf-8',
    )


def write_report_artifacts(
    output_dir: str | Path,
    traces: Sequence[RequestTrace],
    summaries: Sequence[dict[str, Any]],
    mode: str,
    save_raw_requests: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    if save_raw_requests:
        trace_rows = [_trace_to_json(trace) for trace in traces]
        _write_jsonl(output_dir / 'requests.jsonl', trace_rows)
        _write_requests_csv(output_dir / 'requests.csv', trace_rows)
        (output_dir / 'requests.json').write_text(
            json.dumps(trace_rows, indent=2, ensure_ascii=False), encoding='utf-8')
    _write_summary_csv(output_dir / 'summary.csv', summaries)
    (output_dir / 'summary.json').write_text(
        json.dumps(list(summaries), indent=2, ensure_ascii=False), encoding='utf-8')

    plot_paths = [
        path
        for path in [
            _plot_latency_stats(plots_dir, summaries, 'ttft', 'TTFT latency', mode),
            _plot_latency_stats(plots_dir, summaries, 'itl', 'ITL latency', mode),
            _plot_latency_stats(plots_dir, summaries, 'tpot', 'TPOT latency', mode),
            _plot_metric(plots_dir, summaries, 'input_throughput_tok_s', 'Input token throughput', mode),
            _plot_metric(plots_dir, summaries, 'output_throughput_tok_s', 'Output token throughput', mode),
            _plot_metric(plots_dir, summaries, 'success_rate', 'Success rate', mode),
            _plot_token_histogram(plots_dir, traces, 'prompt_tokens', 'Input token histogram'),
            _plot_token_histogram(plots_dir, traces, 'completion_tokens', 'Output token histogram'),
        ]
        if path is not None
    ]
    _write_html_report(output_dir / 'report.html', summaries, plot_paths, mode)


async def _run_warmup(
    requests: Sequence[BenchmarkRequest],
    warmup_requests: int,
    send_one: SendOne,
) -> None:
    if warmup_requests <= 0:
        return
    for request in list(requests)[:warmup_requests]:
        await send_one(request, 'warmup', 0.0, -1)


async def run_benchmark(args: argparse.Namespace) -> tuple[list[RequestTrace], list[dict[str, Any]]]:
    try:
        aiohttp = importlib.import_module('aiohttp')
    except ImportError as e:
        raise RuntimeError('aiohttp is required for live chat-completions benchmarking.') from e

    dataset_files = [Path(path) for path in args.dataset_files] if args.dataset_files else None
    requests = load_requests(
        dataset_dir=args.dataset_dir,
        dataset_files=dataset_files,
        datasets=_split_csv(args.datasets),
        num_prompts=args.num_prompts,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    url = _chat_completions_url(args.base_url, args.api_path)
    headers = {}
    if args.api_key:
        headers['Authorization'] = f"Bearer {args.api_key}"
    extra_body = json.loads(args.extra_request_body) if args.extra_request_body else {}

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
        model = await fetch_model_id(session, args.base_url, args.api_key)
        print(f"Using model from /v1/models: {model}")

        async def send_one(request: BenchmarkRequest, mode: str, setting: float, repeat: int) -> RequestTrace:
            return await request_chat_completion(
                session=session,
                request=request,
                url=url,
                model=model,
                mode=mode,
                setting=setting,
                repeat=repeat,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_completion_tokens=args.max_completion_tokens,
                extra_body=extra_body,
                headers=headers,
                save_response_text=args.save_response_text,
            )

        all_traces: list[RequestTrace] = []
        requests_by_dataset: dict[str, list[BenchmarkRequest]] = defaultdict(list)
        for request in requests:
            requests_by_dataset[request.dataset].append(request)
        for dataset, dataset_requests in sorted(requests_by_dataset.items()):
            print(f"Benchmarking dataset={dataset} requests={len(dataset_requests)}")
            await _run_warmup(dataset_requests, args.warmup_requests, send_one)
            for repeat in range(args.repeats):
                if args.mode == 'concurrency':
                    for concurrency in _parse_number_list(args.levels, as_int=True):
                        all_traces.extend(
                            await closed_loop_runner(
                                dataset_requests,
                                concurrency=int(concurrency),
                                repeat=repeat,
                                send_one=send_one,
                            )
                        )
                elif args.mode == 'request-rate':
                    for request_rate in _parse_number_list(args.levels, as_int=False):
                        all_traces.extend(
                            await request_rate_runner(
                                dataset_requests,
                                request_rate=float(request_rate),
                                repeat=repeat,
                                send_one=send_one,
                                seed=args.seed + repeat,
                            )
                        )

    summaries = aggregate_traces(all_traces)
    write_report_artifacts(
        args.output_dir,
        all_traces,
        summaries,
        mode=args.mode,
        save_raw_requests=args.save_raw_requests,
    )
    return all_traces, summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark /v1/chat/completions with eval JSONL datasets.')
    parser.add_argument(
        '--base-url',
        default='http://127.0.0.1:23333/v1',
        help='OpenAI-compatible API base URL. The model id is fetched from BASE_URL/models.',
    )
    parser.add_argument(
        '--api-path',
        default='',
        help='Chat completions API path. Defaults to /chat/completions when base URL ends with /v1.',
    )
    parser.add_argument('--api-key', default='', help='Bearer token used for /v1/models and chat requests.')
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path('/home/lvhan/lmdeploy/workspace/z1_oc_infer_message_dump'),
        help='Directory containing eval JSONL files. Each file stem is used as the dataset name.',
    )
    parser.add_argument(
        '--dataset-files',
        type=Path,
        nargs='*',
        help='Explicit JSONL files to benchmark. Overrides dataset discovery from --dataset-dir.',
    )
    parser.add_argument(
        '--datasets',
        help='Comma-separated dataset names or filename-stem prefixes, e.g. "bbeh" matches bbeh*.jsonl.',
    )
    parser.add_argument('--num-prompts', type=int, help='Maximum number of prompts sampled per dataset.')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle each dataset before applying --num-prompts.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for shuffling and request-rate scheduling.')
    parser.add_argument(
        '--mode',
        choices=['concurrency', 'request-rate'],
        default='concurrency',
        help='Benchmark mode: closed-loop concurrency or open-loop request rate.',
    )
    parser.add_argument(
        '--levels',
        default='1,2,4,8,16,32,64,128',
        help='Comma-separated sweep values. Interpreted as concurrency levels or request rates based on --mode.',
    )
    parser.add_argument('--repeats', type=int, default=1, help='Number of times to repeat each dataset/level run.')
    parser.add_argument(
        '--warmup-requests',
        type=int,
        default=1,
        help='Number of unmeasured warmup requests to send before each dataset run.',
    )
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature shared by all requests.')
    parser.add_argument('--top-p', type=float, default=1.0, help='Sampling top_p shared by all requests.')
    parser.add_argument('--top-k', type=int, default=40, help='Sampling top_k shared by all requests.')
    parser.add_argument(
        '--max-completion-tokens',
        type=int,
        help='Optional max_completion_tokens limit. If omitted, generation stops naturally at EOS.',
    )
    parser.add_argument(
        '--extra-request-body',
        default='',
        help='JSON object merged into every chat request body for engine-specific options.',
    )
    parser.add_argument(
        '--save-raw-requests',
        action='store_true',
        help='Save per-request raw traces as requests.jsonl, requests.csv, and requests.json.',
    )
    parser.add_argument(
        '--save-response-text',
        action='store_true',
        help='Retain generated and reasoning text in memory and raw traces. Disabled by default to reduce memory use.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('benchmark_outputs') / f"chat_completions_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help='Directory for summary CSV/JSON, PNG plots, HTML report, and optional raw request traces.',
    )
    return parser.parse_args()


def main() -> None:
    traces, summaries = asyncio.run(run_benchmark(parse_args()))
    completed = sum(summary['completed'] for summary in summaries)
    failed = sum(summary['failed'] for summary in summaries)
    print(f"Recorded {len(traces)} requests: {completed} completed, {failed} failed.")


if __name__ == '__main__':
    main()
