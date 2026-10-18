"""Render a README figure directly from an organizer-validation artifact.

Optional tooling: pip install -r requirements-viz.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    rows = [row for row in payload["results"] if row["status"] == "PASS"]
    confirmation = json.loads(args.confirmation.read_text(encoding="utf-8"))
    repeated = [row for row in confirmation["results"] if row["status"] == "PASS"]
    summary = payload["summary"]
    if payload["status"] != "PASS" or len(rows) != 13 or summary["total_failed_elements"]:
        raise ValueError("figure requires the complete, zero-failure 13-row final matrix")
    if (confirmation["status"] != "PASS" or len(repeated) != 13
            or confirmation["summary"]["total_failed_elements"]
            or [row["config"] for row in rows] != [row["config"] for row in repeated]):
        raise ValueError("confirmation must pass the same complete final matrix")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    labels = []
    for index, row in enumerate(rows, 1):
        cfg = row["config"]
        labels.append(f"{index:02d}   B{cfg['batch_size']} · S{cfg['seq_len']} · "
                      f"D{cfg['d_model']} · H{cfg['num_heads']}")
    first = [row["parsed"]["speedup_median"] for row in rows]
    second = [row["parsed"]["speedup_median"] for row in repeated]
    low = [min(a, b) for a, b in zip(first, second)]
    high = [max(a, b) for a, b in zip(first, second)]
    fig, ax = plt.subplots(figsize=(10, 7.2), facecolor="#ffffff")
    fig.subplots_adjust(left=0.33, right=0.95, top=0.78, bottom=0.20)
    bars = ax.barh(range(len(rows)), low, color="#2463b4", height=0.60,
                   xerr=[[0] * len(rows), [b - a for a, b in zip(low, high)]],
                   error_kw={"ecolor": "#172d47", "capsize": 3, "elinewidth": 1.2})
    ax.set_yticks(range(len(rows)), labels, color="#24354b")
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, pad=12)
    ax.spines["bottom"].set_color("#d9e1eb")
    ax.tick_params(axis="x", colors="#53677d", length=0)
    ax.set_xlim(0, max(high) * 1.45)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e9eef4", linewidth=0.8)
    ax.axvline(1, color="#a2aebd", linestyle=(0, (3, 3)), linewidth=1)
    ax.set_xlabel("End-to-end median speedup vs explicit PyTorch attention", labelpad=12,
                  color="#53677d")
    for bar, lower, upper in zip(bars, low, high):
        ax.text(upper + 0.12, bar.get_y() + bar.get_height() / 2,
                f"{lower:.2f}–{upper:.2f}×", va="center", color="#24354b", fontsize=9)
    geomeans = sorted([summary["geometric_mean_speedup"],
                       confirmation["summary"]["geometric_mean_speedup"]])
    fig.text(0.07, 0.92, "Triton Transformer Kernels", fontsize=21, weight="bold",
             color="#172d47")
    fig.text(0.07, 0.865, f"{geomeans[0]:.2f}–{geomeans[1]:.2f}× geometric mean  ·  "
             "13/13 executable rows pass in both runs",
             fontsize=13, color="#2463b4")
    fig.text(0.07, 0.078, "Bars show the lower run; whiskers show the range across "
             "two runs, not a confidence interval.", fontsize=9, color="#53677d")
    fig.text(0.07, 0.048, "RTX 5070 Ti · float32 · five accuracy trials per row · "
             "one authorized resource skip excluded", fontsize=9, color="#53677d")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, metadata={"Date": None})
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=150)
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
