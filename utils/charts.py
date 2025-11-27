from __future__ import annotations

import os
import tempfile
from typing import Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


TEAL = "#2DB7C4"
BLUE = "#3B82F6"
GRID_COLOR = "#D9D9D9"
TEXT_COLOR = "#0F172A"
VALUE_TEXT_COLOR = "#FFFFFF"


def _format_number(value: float) -> str:
    return f"{int(round(value)):,}"


def _setup_axes(ax) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A0A0A0")
    ax.spines["bottom"].set_color("#A0A0A0")
    ax.tick_params(colors="#475569", labelsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}" if x else "0"))


def _add_value_labels(ax, bars, values) -> None:
    max_value = max(values) if values else 0
    if max_value == 0:
        max_value = 1
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if height <= 0:
            continue
        y = height - (0.05 * max_value)
        if y <= 0:
            y = height * 0.5
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            _format_number(val),
            ha="center",
            va="center",
            color=VALUE_TEXT_COLOR,
            fontsize=12,
            fontweight="bold",
        )


def _create_chart(
    filename: str,
    estimated: float,
    actual: float,
    output_dir: str,
) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 12})

    values = [max(estimated or 0, 0), max(actual or 0, 0)]
    labels = ["Estimated", "Actual"]
    colors = [TEAL, BLUE]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    bars = ax.bar(labels, values, color=colors, width=0.55)

    _setup_axes(ax)
    _add_value_labels(ax, bars, values)

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight", transparent=False)
    plt.close(fig)


def generate_summary_charts(
    impressions_actual: float,
    impressions_estimate: float,
    clicks_actual: float,
    clicks_estimate: float,
    output_dir: str = ".",
) -> Tuple[str, str]:
    impressions_path = "impressions_chart.png"
    clicks_path = "clicks_chart.png"

    _create_chart(
        impressions_path,
        impressions_estimate,
        impressions_actual,
        output_dir,
    )

    _create_chart(
        clicks_path,
        clicks_estimate,
        clicks_actual,
        output_dir,
    )

    return (
        os.path.join(output_dir, impressions_path),
        os.path.join(output_dir, clicks_path),
    )
