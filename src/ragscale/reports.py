"""Markdown, HTML, JSON, CSV, and JSONL outputs for one audit."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ragscale.cache import json_safe


def fmt(value: float, percentage: bool = True) -> str:
    if not np.isfinite(value):
        return "not admissible"
    return f"{100 * value:.1f}{'%' if percentage else 'pp'}"


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# ragscale reader-upgrade audit", "",
        f"Dataset: **{summary['dataset']['name']}**  ",
        f"Comparison: **{summary['comparison']['current']} → {summary['comparison']['candidate']}**  ",
        f"Examples: **{summary['dataset']['rows']}**", "",
    ]
    if summary["dataset"].get("description"):
        lines.extend([str(summary["dataset"]["description"]), ""])
    lines += [
        "## Decision summary", "",
        "| Metric | Current compression gain | Raw reader upgrade | Visible compressed upgrade | Upgrade retention | Outcome |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in summary["metrics"]:
        lines.append(
            f"| {result['metric']} | {fmt(result['current_compression_gain'], False)} | "
            f"{fmt(result['raw_upgrade'], False)} | {fmt(result['compressed_upgrade'], False)} | "
            f"{fmt(result['upgrade_retention'])} | {result['outcome']} |"
        )
    for result in summary["metrics"]:
        lines.extend([
            "", f"## {result['metric']} detail", "",
            f"- Raw reader upgrade interval: {fmt(result['raw_upgrade_interval'][0], False)} to {fmt(result['raw_upgrade_interval'][1], False)}",
            f"- Visible compressed upgrade interval: {fmt(result['compressed_upgrade_interval'][0], False)} to {fmt(result['compressed_upgrade_interval'][1], False)}",
            f"- Upgrade-retention interval: {fmt(result['retention_interval'][0])} to {fmt(result['retention_interval'][1])}",
            f"- Configured retention alert: {100 * result['retention_alert_threshold']:.0f}%",
        ])
        if result["current_rescue"] is not None:
            lines.extend([
                f"- Current reader rescue: {fmt(result['current_rescue'])}; raw-correct damage: {fmt(result['current_damage'])}",
                f"- Candidate reader rescue: {fmt(result['candidate_rescue'])}; raw-correct damage: {fmt(result['candidate_damage'])}",
            ])
    lines.extend([
        "", "The deployment and reader decisions are separate. Compression gain measures the current complete pipeline. Upgrade retention measures how much of the observed current-to-candidate reader upgrade remains under the deployed compressor.",
        "", "## Provenance", "",
        f"- Candidate pools: {summary['provenance']['candidate_pools']}",
        f"- Compile-once artifact rows: {summary['provenance']['compressed_artifacts']}",
        f"- Unique compressed content hashes: {summary['provenance']['unique_compressed_artifact_hashes']}",
        f"- Exact reader footprints: {str(summary['provenance']['exact_reader_footprints']).lower()}",
        f"- Raw policy: `{summary['provenance']['raw_policy_id']}`",
    ])
    dataset_hints = summary["dataset"].get("hints") or {}
    if dataset_hints:
        lines.extend(["", "## Dataset hints", ""])
        lines.extend(f"- {key}: {value}" for key, value in dataset_hints.items())
    slices = [row for result in summary["metrics"] for row in result["slices"]]
    if slices:
        lines.extend([
            "", "## Question and dataset slices", "",
            "| Metric | Hint | Value | Rows | Raw upgrade | Compressed upgrade | Retention |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ])
        for row in slices:
            lines.append(
                f"| {row['metric']} | {row['field']} | {row['value']} | {row['rows']} | "
                f"{fmt(row['raw_upgrade'], False)} | {fmt(row['compressed_upgrade'], False)} | {fmt(row['retention'])} |"
            )
    changed = next((result["changed_examples"] for result in summary["metrics"] if result["changed_examples"]), [])
    if changed:
        lines.extend([
            "", "## Changed examples", "",
            "| Example | Type | Current reader | Candidate reader |",
            "| --- | --- | --- | --- |",
        ])
        for row in changed:
            lines.append(
                f"| {row['example_id']} | {row['question_type']} | {row['current_transition']} | {row['candidate_transition']} |"
            )
    resources = summary["resource_summary"]
    lines.extend([
        "", "## Resources", "",
        f"- Mean compressor latency: {resources['mean_compressor_latency_ms']:.1f} ms",
        f"- Mean raw reader latency: {resources['mean_raw_reader_latency_ms']:.1f} ms",
        f"- Mean compressed reader latency: {resources['mean_compressed_reader_latency_ms']:.1f} ms",
        f"- Adapter-reported cost: ${resources['reported_cost_usd']:.4f}",
        f"- Compressor calls this run: {resources['compressor_calls_this_run']}; reader calls this run: {resources['reader_calls_this_run']}; judge calls this run: {resources['judge_calls_this_run']}",
    ])
    lines.extend([
        "", "## Scope", "",
        "This report describes the supplied readers, rows, evidence policy, artifacts, and metrics. It does not certify unseen readers or future compressor draws.",
    ])
    return "\n".join(lines) + "\n"


def html_report(summary: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(result['metric'])}</td>"
        f"<td>{html.escape(fmt(result['current_compression_gain'], False))}</td>"
        f"<td>{html.escape(fmt(result['raw_upgrade'], False))}</td>"
        f"<td>{html.escape(fmt(result['compressed_upgrade'], False))}</td>"
        f"<td>{html.escape(fmt(result['upgrade_retention']))}</td>"
        f"<td><code>{html.escape(result['outcome'])}</code></td>"
        "</tr>"
        for result in summary["metrics"]
    )
    slices = "".join(
        "<tr>"
        f"<td>{html.escape(row['metric'])}</td><td>{html.escape(row['field'])}</td>"
        f"<td>{html.escape(row['value'])}</td><td>{row['rows']}</td>"
        f"<td>{html.escape(fmt(row['raw_upgrade'], False))}</td>"
        f"<td>{html.escape(fmt(row['compressed_upgrade'], False))}</td>"
        f"<td>{html.escape(fmt(row['retention']))}</td></tr>"
        for result in summary["metrics"] for row in result["slices"]
    )
    resources = summary["resource_summary"]
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ragscale audit</title>
<style>
body{{font:15px/1.5 system-ui;color:#18212f;max-width:1100px;margin:40px auto;padding:0 24px}}
h1,h2{{letter-spacing:-.02em}} table{{border-collapse:collapse;width:100%;margin:16px 0 28px}}
th,td{{border-bottom:1px solid #d8dee8;padding:9px;text-align:left}} th{{background:#f3f6fa}}
code{{background:#eef2f7;padding:2px 5px;border-radius:4px}} .scope{{background:#f6f8fa;padding:16px;border-radius:8px}}
</style></head><body>
<h1>ragscale reader-upgrade audit</h1>
<p><strong>{html.escape(summary['dataset']['name'])}</strong>, {summary['dataset']['rows']} examples<br>
{html.escape(summary['comparison']['current'])} → {html.escape(summary['comparison']['candidate'])}</p>
<p>{html.escape(str(summary['dataset'].get('description') or ''))}</p>
<h2>Decision summary</h2>
<table><thead><tr><th>Metric</th><th>Current compression gain</th><th>Raw reader upgrade</th>
<th>Visible compressed upgrade</th><th>Retention</th><th>Outcome</th></tr></thead><tbody>{rows}</tbody></table>
<p>Deployment utility and reader-upgrade fidelity are separate outcomes.</p>
<h2>Question and dataset slices</h2>
<table><thead><tr><th>Metric</th><th>Hint</th><th>Value</th><th>Rows</th><th>Raw upgrade</th>
<th>Compressed upgrade</th><th>Retention</th></tr></thead><tbody>{slices}</tbody></table>
<h2>Provenance and resources</h2>
<ul><li>{summary['provenance']['candidate_pools']} candidate pools</li>
<li>{summary['provenance']['compressed_artifacts']} compile-once artifact rows</li>
<li>{summary['provenance']['unique_compressed_artifact_hashes']} unique compressed content hashes</li>
<li>Exact reader footprints: {str(summary['provenance']['exact_reader_footprints']).lower()}</li>
<li>Mean compressor latency: {resources['mean_compressor_latency_ms']:.1f} ms</li>
<li>Adapter-reported cost: ${resources['reported_cost_usd']:.4f}</li></ul>
<p class="scope">{html.escape(summary['scope'])}</p>
</body></html>\n"""


def write_outputs(summary: dict[str, Any], frame: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(markdown_report(summary), encoding="utf-8")
    (output / "report.html").write_text(html_report(summary), encoding="utf-8")
    frame.to_csv(output / "paired_scores.csv", index=False)
    with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in frame.to_dict(orient="records"):
            handle.write(json.dumps(json_safe(record), sort_keys=True) + "\n")
