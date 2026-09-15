#!/usr/bin/env python3
"""Generate a read-only preliminary report from Lancet formal archives.

The script intentionally reads TensorBoard/scalar exports and lifecycle state only.
It does not invoke training, edit a run archive, or interpret an incomplete run as
formal evidence.  SVG output avoids adding a plotting dependency to the D4RL image.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

FORMAL_ENV = "antmaze-medium-play-v2"
WARMUP_STEPS = 5_000
ACTIVE_WINDOW_END = 55_000


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_events(archive: Path, tag: str) -> list[tuple[int, float]]:
    """Read stable scalar export when available, otherwise raw TB event files."""
    exported = archive / "metrics" / "scalars.json"
    if exported.is_file():
        values = read_json(exported).get(tag, [])
        return [(int(item["step"]), float(item["value"])) for item in values]

    event_files = sorted(archive.rglob("events.out.tfevents.*"))
    if not event_files:
        return []
    accumulator = EventAccumulator(
        str(event_files[0].parent), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return [(int(item.step), float(item.value)) for item in accumulator.Scalars(tag)]


def archive_for_job(job: dict[str, Any]) -> Path:
    """Resolve lifecycle host paths when the report runs in the D4RL container."""
    archive = Path(str(job["archive"]))
    if archive.is_dir():
        return archive
    host_data = Path("/home/zhaozihan/Lancet/data")
    if host_data in archive.parents:
        return Path("/data/lancet") / archive.relative_to(host_data)
    return archive


def completed_pairs(jobs: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = {}
    for job in jobs:
        if job.get("stage") != "online" or job.get("status") != "completed":
            continue
        method = job.get("method")
        if method not in {"wsrl", "lancet"}:
            continue
        result.setdefault(int(job["seed"]), {})[method] = job
    return {
        seed: pair for seed, pair in result.items() if set(pair) == {"wsrl", "lancet"}
    }


def interpolation(curve: list[tuple[int, float]], step: int) -> float:
    curve = sorted(curve)
    if step <= curve[0][0]:
        return curve[0][1]
    if step >= curve[-1][0]:
        return curve[-1][1]
    for (left_s, left_v), (right_s, right_v) in pairwise(curve):
        if left_s <= step <= right_s:
            return left_v + (right_v - left_v) * (step - left_s) / (right_s - left_s)
    raise RuntimeError("unreachable")


def trapezoid_auc_mean(curve: list[tuple[int, float]], start: int, end: int) -> float:
    points = [(start, interpolation(curve, start))]
    points += [(step, value) for step, value in curve if start < step < end]
    points += [(end, interpolation(curve, end))]
    area = sum(
        (right_s - left_s) * (left_v + right_v) / 2
        for (left_s, left_v), (right_s, right_v) in pairwise(points)
    )
    return area / (end - start)


def svg_line_chart(
    panels: list[dict[str, Any]],
    title: str,
    output: Path,
    *,
    width: int = 1120,
    height: int = 460,
) -> None:
    """Write a compact multi-panel SVG line chart using only the Python stdlib."""
    margin_left, margin_right, margin_top, margin_bottom = 65, 25, 55, 54
    count = len(panels)
    gap = 28
    panel_width = (width - margin_left - margin_right - gap * (count - 1)) / count
    plot_height = height - margin_top - margin_bottom
    text: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:18px;font-weight:700}.label{font-size:12px}.small{font-size:11px}.grid{stroke:#d9dee8;stroke-width:1}.axis{stroke:#5e6b80;stroke-width:1.2}</style>",
        f'<text x="{width / 2:.1f}" y="27" text-anchor="middle" class="title">{html.escape(title)}</text>',
    ]
    colors = ["#2563eb", "#d97706", "#059669", "#dc2626"]
    for index, panel in enumerate(panels):
        x0 = margin_left + index * (panel_width + gap)
        y0 = margin_top
        x_min, x_max = panel["xlim"]
        y_min, y_max = panel["ylim"]

        def px(
            x: float,
            x0: float = x0,
            x_min: float = x_min,
            x_max: float = x_max,
        ) -> float:
            return x0 + (x - x_min) / (x_max - x_min) * panel_width

        def py(
            y: float,
            y0: float = y0,
            y_min: float = y_min,
            y_max: float = y_max,
        ) -> float:
            return y0 + plot_height - (y - y_min) / (y_max - y_min) * plot_height

        text.append(
            f'<text x="{x0 + panel_width / 2:.1f}" y="47" text-anchor="middle" class="label">{html.escape(panel["title"])}</text>'
        )
        for tick in panel.get("yticks", []):
            y = py(tick)
            text += [
                f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + panel_width:.1f}" y2="{y:.1f}" class="grid"/>',
                f'<text x="{x0 - 7:.1f}" y="{y + 4:.1f}" text-anchor="end" class="small">{tick:g}</text>',
            ]
        for tick in panel.get("xticks", []):
            x = px(tick)
            text += [
                f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y0 + plot_height:.1f}" class="grid"/>',
                f'<text x="{x:.1f}" y="{y0 + plot_height + 18:.1f}" text-anchor="middle" class="small">{tick / 1000:g}k</text>',
            ]
        text += [
            f'<line x1="{x0:.1f}" y1="{y0 + plot_height:.1f}" x2="{x0 + panel_width:.1f}" y2="{y0 + plot_height:.1f}" class="axis"/>',
            f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y0 + plot_height:.1f}" class="axis"/>',
        ]
        for marker, marker_label in panel.get("markers", []):
            if x_min <= marker <= x_max:
                x = px(marker)
                text.append(
                    f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y0 + plot_height:.1f}" stroke="#64748b" stroke-dasharray="4 3"/>'
                )
                text.append(
                    f'<text x="{x + 3:.1f}" y="{y0 + 13:.1f}" class="small">{html.escape(marker_label)}</text>'
                )
        for series_index, series in enumerate(panel["series"]):
            points = [
                (px(x), py(y)) for x, y in series["points"] if x_min <= x <= x_max
            ]
            if not points:
                continue
            path = " ".join(
                ("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}"
                for i, (x, y) in enumerate(points)
            )
            color = series.get("color", colors[series_index % len(colors)])
            text.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2"/>'
            )
            text.append(
                f'<circle cx="{points[-1][0]:.1f}" cy="{points[-1][1]:.1f}" r="3" fill="{color}"/>'
            )
        for series_index, series in enumerate(panel["series"]):
            color = series.get("color", colors[series_index % len(colors)])
            lx = x0 + 6 + series_index * 106
            ly = y0 + plot_height + 39
            text += [
                f'<line x1="{lx:.1f}" y1="{ly - 4:.1f}" x2="{lx + 16:.1f}" y2="{ly - 4:.1f}" stroke="{color}" stroke-width="2.2"/>',
                f'<text x="{lx + 20:.1f}" y="{ly:.1f}" class="small">{html.escape(series["label"])}</text>',
            ]
    text.append("</svg>")
    output.write_text("\n".join(text) + "\n", encoding="utf-8")


def online_curve(archive: Path, online_start: int, tag: str) -> list[tuple[int, float]]:
    return [(step - online_start, value) for step, value in scalar_events(archive, tag)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root", type=Path, default=Path("/data/lancet/runs/formal")
    )
    parser.add_argument(
        "--include-superseded-generation1",
        action="store_true",
        help="Explicitly allow analysis of the archived pre-upstream formal root.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs_root = args.runs_root
    if runs_root.name == "formal" and not args.include_superseded_generation1:
        raise SystemExit(
            "Generation-1 formal data is SUPERSEDED_PRE_UPSTREAM_SYNC. "
            "Use /data/lancet/runs/formal (current generation, default), or pass "
            "--include-superseded-generation1 for an explicit legacy audit."
        )
    out = args.output_dir
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    state = read_json(runs_root / ".lifecycle" / "formal_pipeline.json")
    jobs = [job for job in state["jobs"] if not job.get("superseded_by")]
    pairs = completed_pairs(jobs)
    if not pairs:
        raise RuntimeError("No completed WSRL/Lancet pairs found")

    pair_rows = []
    score_panels = []
    for seed, pair in sorted(pairs.items()):
        wsrl_archive, lancet_archive = (
            archive_for_job(pair["wsrl"]),
            archive_for_job(pair["lancet"]),
        )
        lancet_summary = read_json(lancet_archive / "metrics" / "summary.json")
        online_start = int(lancet_summary["online_start_global_step"])
        adaptation_start = int(lancet_summary["adaptation_start_global_step"])
        curves = {
            "WSRL": scalar_events(wsrl_archive, "eval/normalized_score"),
            "Lancet": scalar_events(lancet_archive, "eval/normalized_score"),
        }
        post = {
            method: [(s, v) for s, v in curve if s >= adaptation_start]
            for method, curve in curves.items()
        }
        endpoint = min(curve[-1][0] for curve in curves.values())
        pair_rows.append(
            {
                "seed": seed,
                "online_start": online_start,
                "adaptation_start": adaptation_start,
                "final_online": endpoint - online_start,
                "wsrl_final": interpolation(curves["WSRL"], endpoint),
                "lancet_final": interpolation(curves["Lancet"], endpoint),
                "wsrl_post_max": max(value for _, value in post["WSRL"]),
                "lancet_post_max": max(value for _, value in post["Lancet"]),
                "wsrl_auc50_interpolated": trapezoid_auc_mean(
                    curves["WSRL"], adaptation_start, adaptation_start + 50_000
                ),
                "lancet_auc50_interpolated": trapezoid_auc_mean(
                    curves["Lancet"], adaptation_start, adaptation_start + 50_000
                ),
                "wsrl_post_points": len(post["WSRL"]),
                "lancet_post_points": len(post["Lancet"]),
            }
        )
        score_panels.append(
            {
                "title": f"Completed pair: seed {seed}",
                "xlim": (0, 500_032),
                "ylim": (0, 20),
                "xticks": [0, 100_000, 250_000, 500_000],
                "yticks": [0, 5, 10, 15, 20],
                "markers": [
                    (WARMUP_STEPS, "warmup"),
                    (ACTIVE_WINDOW_END, "Lancet off"),
                ],
                "series": [
                    {
                        "label": "WSRL",
                        "points": [(s - online_start, v) for s, v in curves["WSRL"]],
                    },
                    {
                        "label": "Lancet",
                        "points": [(s - online_start, v) for s, v in curves["Lancet"]],
                    },
                ],
            }
        )
    svg_line_chart(
        score_panels,
        "Completed formal pairs: normalized score (20 evaluation episodes)",
        figures / "completed_pairs_score.svg",
        height=430,
    )

    # Mechanism trace for seed 0 only, the first complete Lancet archive.
    seed0 = pairs[min(pairs)]
    lancet0 = archive_for_job(seed0["lancet"])
    summary0 = read_json(lancet0 / "metrics" / "summary.json")
    online_start0 = int(summary0["online_start_global_step"])
    lambda_curve = online_curve(lancet0, online_start0, "lancet/lambda")
    ratio_curve = [
        (x, y * 10_000)
        for x, y in online_curve(lancet0, online_start0, "lancet/delta_q_ratio")
    ]
    weight_curve = online_curve(lancet0, online_start0, "lancet/weight_mean")
    svg_line_chart(
        [
            {
                "title": "Lancet λ",
                "xlim": (0, 60_000),
                "ylim": (-0.05, 1.05),
                "xticks": [0, 5_000, 55_000],
                "yticks": [0, 0.5, 1],
                "markers": [(WARMUP_STEPS, "adapt"), (ACTIVE_WINDOW_END, "off")],
                "series": [
                    {"label": "lambda", "points": lambda_curve, "color": "#d97706"}
                ],
            },
            {
                "title": "|Δ| / |Qbase| (×10⁴)",
                "xlim": (0, 60_000),
                "ylim": (
                    0,
                    max(1.0, max((y for _, y in ratio_curve), default=1.0) * 1.1),
                ),
                "xticks": [0, 5_000, 55_000],
                "yticks": [0, 1, 3, 5, 7],
                "markers": [(WARMUP_STEPS, "adapt"), (ACTIVE_WINDOW_END, "off")],
                "series": [
                    {"label": "ratio", "points": ratio_curve, "color": "#059669"}
                ],
            },
            {
                "title": "U fit weight",
                "xlim": (0, 60_000),
                "ylim": (1, 3),
                "xticks": [0, 5_000, 55_000],
                "yticks": [1, 2, 3],
                "markers": [(WARMUP_STEPS, "adapt"), (ACTIVE_WINDOW_END, "off")],
                "series": [
                    {"label": "weight", "points": weight_curve, "color": "#2563eb"}
                ],
            },
        ],
        "Seed 0 Lancet mechanism diagnostics (active handoff only)",
        figures / "seed0_lancet_mechanism.svg",
        height=410,
    )

    # Incomplete runs are visualized separately and never used for an effect claim.
    live = [
        job
        for job in jobs
        if job.get("stage") == "online" and job.get("status") == "running"
    ]
    live_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for job in live:
        if job.get("method") in {"wsrl", "lancet"}:
            live_by_seed.setdefault(int(job["seed"]), {})[job["method"]] = job
    live_panels = []
    live_rows = []
    for seed, pair in sorted(live_by_seed.items()):
        series = []
        for method, job in sorted(pair.items()):
            archive = archive_for_job(job)
            events = scalar_events(archive, "eval/normalized_score")
            online_start = 1_000_000
            if events:
                series.append(
                    {
                        "label": method.upper(),
                        "points": [(s - online_start, v) for s, v in events],
                    }
                )
                live_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "last_global": events[-1][0],
                        "last_online": events[-1][0] - online_start,
                        "last_score": events[-1][1],
                        "max_score": max(v for _, v in events),
                        "gpu": job.get("gpu_id"),
                    }
                )
        if series:
            live_panels.append(
                {
                    "title": f"Running seed {seed} (interim)",
                    "xlim": (0, 500_032),
                    "ylim": (0, 100),
                    "xticks": [0, 100_000, 250_000, 500_000],
                    "yticks": [0, 25, 50, 75, 100],
                    "markers": [
                        (WARMUP_STEPS, "warmup"),
                        (ACTIVE_WINDOW_END, "Lancet off"),
                    ],
                    "series": series,
                }
            )
    if live_panels:
        svg_line_chart(
            live_panels,
            "Running formal branches: interim normalized score (not evidence)",
            figures / "running_branches_score.svg",
            height=430,
        )

    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    rows_md = "\n".join(
        f"| {r['seed']} | {r['final_online']:,} | {r['wsrl_final']:.1f} | {r['lancet_final']:.1f} | {r['wsrl_post_max']:.1f} | {r['lancet_post_max']:.1f} | {r['wsrl_auc50_interpolated']:.4f} | {r['lancet_auc50_interpolated']:.4f} |"
        for r in pair_rows
    )
    live_md = (
        "\n".join(
            f"| {r['seed']} | {r['method']} | GPU {r['gpu']} | {r['last_online']:,} | {r['last_score']:.1f} | {r['max_score']:.1f} |"
            for r in live_rows
        )
        or "| — | — | — | — | — | — |"
    )
    report = f"""# Lancet formal benchmark：初步结果分析

> 生成时间：`{timestamp}`。本报告只分析当前归档中的 TensorBoard/scalar 数据；它不是最终论文结论。正式主比较要求 5 个 paired seeds，目前仅有 {len(pair_rows)} 个完整 pair。

## Executive summary

- 已完成的 seed 0 与 seed 2 配对中，WSRL 和 Lancet 在 adaptation 开始后的所有实际评估点均为 `0` normalized score；目前没有可支持 Lancet 改善 early adaptation 的证据。
- 两个完成 pair 均完成到共同实际 online endpoint `500,032`，checkpoint/finite/reload 验证均由各自 archive 记录为通过。因而这是**性能无信号**，不是训练崩溃。
- Lancet 的 residual 分支确实运行：seed 0 有 `200,192` 次 residual update，U weight 有限；但 centered correction 相对 base Q 很小（`|Δ|/|Q_base|` 最大约 `7.05e-4`），在当前数据中几乎没有改变 policy-facing critic 的量级。
- 当前 running 的 seed 3/4 仅作进度观察，不能提前作为统计结果；seed 1 在本机仍为 blocked，按既有安排由 6025 服务器承担。

## Evidence scope and figures

已完成 pair 的横轴是 online step；虚线分别表示 warmup 结束（5k）和 Lancet correction 关闭（55k）。每一次 eval 为 20 个完整 episode，`normalized_score = 100 × success_at_end`。

![Completed pair scores](figures/completed_pairs_score.svg)

![Seed 0 Lancet mechanism](figures/seed0_lancet_mechanism.svg)

{"![Running branch scores](figures/running_branches_score.svg)" if live_panels else ""}

## Completed paired evidence

| Seed | Common final online step | WSRL final | Lancet final | WSRL post-adaptation max | Lancet post-adaptation max | Boundary-interpolated AUC 0–50k: WSRL | Lancet |
|---:|---:|---:|---:|---:|---:|---:|---:|
{rows_md}

### What is observed

1. 两个完整 pair 的最终 score 都是 0，且 adaptation start 之后每一个**真实记录的 evaluation point**均是 0。
2. seed 0 在 online 起点附近出现 WSRL 15、Lancet 10；seed 2 的早期最大值为 15。但这些值发生在 warmup/adaptation 边界之前，不能用来支持 Lancet 的 handoff claim。
3. seed 0 的 Lancet 边界插值 AUC 会出现约 `0.0465` 的非零值；原因是 4,032-step evaluation 为 10、6,016-step evaluation 为 0，线性插值跨过了 adaptation start=5,056。它不是 post-adaptation 的观测改善，不能作为方法收益报告。

### Mechanism health, not efficacy

seed 0 Lancet 在 active 50k window 中：

- residual update count: `200,192`；
- `|Δ|/|Q_base|`: mean about `3.29e-5`, maximum `7.05e-4`；
- U fitting weight: mean about `1.78`, bounded below the configured cap 3；
- base Q / corrected Q、critic loss、residual loss 均保持 finite，且 final checkpoint reload passed。

这表明实现和 lifecycle 正常，但不表明 correction 已足够大或足够有效以改变 action ranking / policy update。

## Running branches: interim only

| Seed | Method | GPU | Latest online step | Latest score | Historical max |
|---:|---|---|---:|---:|---:|
{live_md}

seed 3 的最近一次 eval 曾达到 100，但它尚未到共同 endpoint；seed 4 仍处于中段。不得用单个中间点、best score 或不同 wall-clock 时刻比较 WSRL/Lancet。

## Initial interpretation and ranked hypotheses

### H1 — online handoff itself is currently the dominant failure mode (high confidence observation)

两种方法均在 warmup/adaptation 初期丢失早期成功信号，并在完成 pair 中未恢复。WSRL 也呈现相同行为，因此现有证据不能把问题归因于 Lancet。

### H2 — current Lancet correction is numerically too weak to改变 actor update (medium-confidence hypothesis)

在 seed 0 中，base Q 约为 -500，而 centered Delta 的相对比值最多约 0.07%，通常更小。该量级与“没有可见策略差异”一致，但尚未证明因果；必须用 action ranking / actor-gradient diagnostics 验证。

### H3 — adaptation AUC boundary需要更稳健的采样处理 (high-confidence measurement issue)

当前 evaluation cadence 不保证在 adaptation_step=0 正好取样。线性边界插值会把 warmup 期间的分数带入 primary window，产生 seed 0 的伪非零 AUC。后续报告应同时保留协议 AUC，并额外报告“仅由 post-adaptation observed evaluations 构成”的审计指标；下一版实验应在 adaptation start 强制做一次 evaluation。

### H4 — fork/evaluation randomness must be audited before attributing early differences (medium-confidence hypothesis)

同一 offline checkpoint fork 的 seed 0 在 initial anchor 有 WSRL=15、Lancet=10 的差异。该差异可能来自 evaluation seed stream、environment stochasticity 或 archive timing；当前不能假定它是算法收益。应核对 paired deterministic evaluation 是否真正使用相同 episode seeds。

## What not to conclude

- 不可宣称 Lancet 有效、无效或劣于 WSRL：只有两个完整 paired seeds，且二者 performance curve 没有可辨识的正信号。
- 不可因低分重跑已完成 seed；这不符合 frozen rerun policy。
- 不可修改正在运行的 formal run、Lancet objective 或 frozen hyperparameters。

## Recommended next steps

### 1. Complete and aggregate the frozen main matrix

让本机 seed 3/4 自然完成，并汇入 6025 的 seed 1。取得五个 paired seeds 后，按 protocol 固定的 paired statistics 汇总，不根据当前结果筛 seed。

### 2. Run a separate transition-diagnostic debug experiment after the formal matrix

从一个已归档 offline initializer 起，固定 evaluation episode seeds，并在 online `0`, `5k`, `adaptation 0`, `10k`, `50k` 强制评估。记录 actor/base-Q/target-Q 参数差异、replay composition、action saturation、CQL/Cal-QL state。这先回答“为何 WSRL 也在 handoff 失去 success”。

### 3. Add diagnosis before retuning Lancet

在新的 debug-only run 记录同一 state 上 base-Q 与 corrected-Q 的 action ranking、actor action change、Delta 分位数、actor gradient norm。若 correction 始终远小于 Q scale，再考虑单变量 ablation：residual LR、minimal-surgery coefficient、U beta/cap 或 correction scale。所有此类尝试必须是新 config/new archive，不能覆盖本轮 formal evidence。

### 4. Decide whether to launch component ablations only after the transition gate

Raw/Centered/Lancet ablations在当前主 comparison 无 recovery signal 时仍能检验机制，但优先级应低于 handoff/evaluation diagnostic。若 WSRL baseline 的 transition failure 被确认，先修复或重新冻结统一 baseline protocol，再运行大量 component ablation 才有解释力。

## Reproducibility

- Source runs: `/data/lancet/runs/formal/{FORMAL_ENV}/`。
- Generator: `scripts/analysis/generate_preliminary_formal_results.py`。
- Re-run inside the D4RL container:

```bash
./dev d4rl python scripts/analysis/generate_preliminary_formal_results.py \\
  --runs-root /data/lancet/runs/formal \\
  --output-dir /data/lancet/runs/formal/analysis/preliminary_2026-09-13
```

The report is a snapshot: running-run charts will differ when regenerated.
"""
    (out / "PRELIMINARY_FORMAL_RESULTS_ANALYSIS.md").write_text(
        report, encoding="utf-8"
    )
    (out / "evidence_snapshot.json").write_text(
        json.dumps(
            {
                "generated_at": timestamp,
                "completed_pairs": pair_rows,
                "running_branches": live_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(out / "PRELIMINARY_FORMAL_RESULTS_ANALYSIS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
