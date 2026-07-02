"""Robustness heatmap — the signature ``type × severity × success-rate`` figure.

One panel per controller so the graduated matrix can be read at a glance and
compared fairly (identical cells, identical seeds). Each panel is a
disturbance-type × severity grid coloured by success rate. Because severity
scales differ per type (wind m/s, signed mass fraction, sensor-noise σ), the
x-axis is an *ordinal* level and each cell is annotated with its actual severity
so nothing is hidden. Wind aggregates over direction and sensor noise over spike
probability by mean success rate at each level; the full per-direction /
per-spike detail stays in ``robustness_matrix.csv``.

Honesty: per-cell sample counts are printed in the figure caption (a small n can
make a success rate look more precise than it is — see the Risk Register).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: file output only, no display backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)

# Disturbance types shown as heatmap rows, in a fixed, readable order. "nominal"
# is not a row — it is reported as the panel's baseline reference instead.
_TYPE_ORDER: tuple[str, ...] = ("wind", "mass", "sensor_noise", "combined")
_TYPE_LABELS = {
    "wind": "wind (m/s)",
    "mass": "mass offset",
    "sensor_noise": "sensor noise (σ)",
    "combined": "combined",
}


def _aggregate_by_type(
    rows: list[dict[str, object]], controller: str
) -> dict[str, list[tuple[float, float]]]:
    """Return ``{type: [(severity, mean_success_rate), ...]}`` for one controller.

    Averages success rate over the secondary axis (wind direction, spike
    probability) at each severity level, then sorts levels ascending.
    """
    acc: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in rows:
        if r["controller"] != controller or r["disturbance_type"] == "nominal":
            continue
        key = (str(r["disturbance_type"]), float(r["severity"]))
        acc[key].append(float(r["success_rate"]))

    per_type: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (dtype, severity), vals in acc.items():
        per_type[dtype].append((severity, sum(vals) / len(vals)))
    for dtype in per_type:
        per_type[dtype].sort(key=lambda pair: pair[0])
    return per_type


def _nominal_success(rows: list[dict[str, object]], controller: str) -> float | None:
    """Return the controller's nominal (baseline) success rate, if present."""
    for r in rows:
        if r["controller"] == controller and r["disturbance_type"] == "nominal":
            return float(r["success_rate"])
    return None


def _draw_panel(ax, rows: list[dict[str, object]], controller: str) -> object:
    """Draw one controller's type × severity heatmap; return the image handle."""
    per_type = _aggregate_by_type(rows, controller)
    types = [t for t in _TYPE_ORDER if t in per_type]
    max_levels = max((len(per_type[t]) for t in types), default=1)

    grid = np.full((len(types), max_levels), np.nan)
    for i, dtype in enumerate(types):
        for j, (_severity, success) in enumerate(per_type[dtype]):
            grid[i, j] = success

    im = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")

    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([_TYPE_LABELS.get(t, t) for t in types])
    ax.set_xticks(range(max_levels))
    ax.set_xticklabels([f"L{j}" for j in range(max_levels)])
    ax.set_xlabel("severity level")

    # Annotate each populated cell with success rate + the actual severity.
    for i, dtype in enumerate(types):
        for j, (severity, success) in enumerate(per_type[dtype]):
            ax.text(
                j, i, f"{success:.0%}\n@{severity:g}",
                ha="center", va="center", fontsize=7, color="black",
            )

    nominal = _nominal_success(rows, controller)
    title = controller.upper()
    if nominal is not None:
        title += f"\nnominal: {nominal:.0%}"
    ax.set_title(title, fontsize=10)
    return im


def plot_robustness_heatmap(
    rows: list[dict[str, object]],
    out_path: str | Path,
    episodes_per_cell: int | None = None,
) -> Path:
    """Render the per-controller robustness heatmap grid to ``out_path``.

    Parameters
    ----------
    rows : list of dict
        Long-format matrix rows from :func:`robustness.evaluation.run_matrix`.
    out_path : str or Path
        Destination PNG path.
    episodes_per_cell : int, optional
        Sample count per cell, printed in the caption for honesty. When omitted
        it is inferred from the rows' ``n_episodes``.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    controllers = sorted({str(r["controller"]) for r in rows})
    if episodes_per_cell is None and rows:
        episodes_per_cell = int(max(int(r["n_episodes"]) for r in rows))

    fig, axes = plt.subplots(
        1, len(controllers), figsize=(1 + 3.2 * len(controllers), 4.2), squeeze=False
    )
    im = None
    for ax, controller in zip(axes[0], controllers):
        im = _draw_panel(ax, rows, controller)

    if im is not None:
        cbar = fig.colorbar(im, ax=list(axes[0]), fraction=0.046, pad=0.04)
        cbar.set_label("success rate")

    fig.suptitle("Robustness matrix — success rate by disturbance type × severity")
    caption = "graduated fixed-seed matrix; identical conditions per cell"
    if episodes_per_cell:
        caption += f"; n = {episodes_per_cell} episodes/cell"
    fig.text(0.5, 0.01, caption, ha="center", fontsize=8, style="italic")

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)
    return out
