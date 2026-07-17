"""Per-controller robustness cards — the within-controller deployment verdict.

A *card* answers the mission question for one controller: "is this controller
robust enough to deploy, and at what disturbance magnitude does it break?" It
plots a degradation curve (success rate vs severity) per disturbance family and
reports a **break-point**: the smallest tested severity whose mean success rate
falls below a configurable deployment gate. Variants of the same controller
(e.g. nominal-trained vs robust-trained) share one card so the effect of
robustness training is read directly — same architecture, same cells, same
seeds; the training distribution is the only delta.

Cards complement (never replace) the cross-controller heatmap
(:mod:`robustness.heatmap`): a break-point is only interpretable against a
reference controller, and the graduated fixed-seed matrix remains the fairness
anchor. Both consume the same long-format matrix rows.

Aggregation follows the heatmap's conventions: wind averages over direction at
each magnitude and sensor noise averages over spike probability at each σ (the
σ=0 level therefore includes spike-only cells); the full per-cell detail stays
in the matrix CSV. Mass offset keeps its sign — the curve runs from the
heaviest negative offset through nominal (0) to the heaviest positive one, and
its break-point is the smallest *magnitude* that fails on either side.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: file output only, no display backend
import matplotlib.pyplot as plt  # noqa: E402

from utils.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)

# Disturbance families drawn as degradation-curve panels, in reading order.
# "combined" is a single max-severity cell, reported as a headline value
# rather than a curve; "nominal" anchors the curves at severity 0.
FAMILY_ORDER: tuple[str, ...] = ("wind", "mass", "sensor_noise")
_FAMILY_LABELS = {
    "wind": "wind (m/s)",
    "mass": "mass offset (fraction)",
    "sensor_noise": "sensor noise (σ)",
}
# Families whose severity axis is signed (curve crosses nominal at 0).
_SIGNED_FAMILIES: frozenset[str] = frozenset({"mass"})

# Fixed variant styling, assigned in declaration order and never cycled.
# Colour pair is Paul Tol's colourblind-safe blue/orange; identity is also
# carried by linestyle + marker so it never rides on colour alone.
_VARIANT_COLORS: tuple[str, ...] = ("#4477AA", "#EE7733", "#228833", "#AA3377")
_VARIANT_LINESTYLES: tuple[str, ...] = ("-", "--", "-.", ":")
_VARIANT_MARKERS: tuple[str, ...] = ("o", "s", "^", "D")


def load_matrix_rows(csv_path: str | Path) -> list[dict[str, object]]:
    """Load long-format matrix rows written by ``robustness.evaluation``.

    Returns the rows as dicts (strings preserved; numeric fields parsed where
    the card computations need them). A missing file returns an empty list so
    callers can merge whichever matrix CSVs exist so far.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning("matrix csv %s not found — skipping", path)
        return []
    with path.open(newline="") as fh:
        rows: list[dict[str, object]] = list(csv.DictReader(fh))
    logger.info("loaded %d matrix rows from %s", len(rows), path)
    return rows


def _controller_rows(
    rows: list[dict[str, object]], controller: str
) -> list[dict[str, object]]:
    return [r for r in rows if str(r["controller"]) == controller]


def nominal_success(rows: list[dict[str, object]], controller: str) -> float | None:
    """Return the controller's nominal (undisturbed) success rate, if present."""
    for r in _controller_rows(rows, controller):
        if str(r["disturbance_type"]) == "nominal":
            return float(str(r["success_rate"]))
    return None


def combined_success(rows: list[dict[str, object]], controller: str) -> float | None:
    """Return the combined-max-severity cell's success rate, if present."""
    for r in _controller_rows(rows, controller):
        if str(r["disturbance_type"]) == "combined":
            return float(str(r["success_rate"]))
    return None


def degradation_curve(
    rows: list[dict[str, object]], controller: str, family: str
) -> list[tuple[float, float]]:
    """Return ``[(severity, mean success rate), ...]`` sorted by severity.

    Success is averaged over the family's secondary axis (wind direction,
    spike probability) at each severity level. The nominal cell is included as
    the severity-0 anchor unless the family already has a 0 level (sensor
    noise tests σ=0 with spikes, which is *not* the nominal condition).
    """
    acc: dict[float, list[float]] = defaultdict(list)
    for r in _controller_rows(rows, controller):
        if str(r["disturbance_type"]) != family:
            continue
        acc[float(str(r["severity"]))].append(float(str(r["success_rate"])))

    curve = [(sev, sum(vals) / len(vals)) for sev, vals in acc.items()]
    zero_level_tested = any(sev == 0.0 for sev, _ in curve)
    nominal = nominal_success(rows, controller)
    if curve and nominal is not None and not zero_level_tested:
        curve.append((0.0, nominal))
    curve.sort(key=lambda pair: pair[0])
    return curve


def break_point(
    curve: list[tuple[float, float]], gate: float
) -> tuple[float | None, float | None]:
    """Return ``(break_severity, max_tested)`` for one degradation curve.

    ``break_severity`` is the smallest tested severity *magnitude* whose mean
    success rate falls below ``gate`` (signed families break on whichever side
    fails first). ``None`` means the gate held everywhere tested — a
    right-censored result: the true break-point lies beyond ``max_tested``,
    which is reported so "held to X" is never read as "unbreakable".
    """
    if not curve:
        return None, None
    max_tested = max(abs(sev) for sev, _ in curve)
    failing = sorted(abs(sev) for sev, success in curve if success < gate)
    return (failing[0] if failing else None), max_tested


def build_card_summary(
    rows: list[dict[str, object]],
    card: str,
    variants: dict[str, str],
    gate: float,
) -> dict[str, object]:
    """Assemble one card's serializable summary (curves + break-points).

    Parameters
    ----------
    rows : list of dict
        Merged long-format matrix rows (any number of CSVs).
    card : str
        Card name, e.g. ``"sac"``.
    variants : dict of str -> str
        ``{variant label: controller name in the rows}``. Variants absent from
        the rows are skipped with a warning so cards can be built before every
        variant has been trained/evaluated.
    gate : float
        Deployment gate on success rate defining the break-point.
    """
    present = {str(r["controller"]) for r in rows}
    summary_variants: dict[str, object] = {}
    for label, controller in variants.items():
        if controller not in present:
            logger.warning(
                "card %s: variant %r (controller %r) not in matrix rows — skipping",
                card,
                label,
                controller,
            )
            continue
        families: dict[str, object] = {}
        for family in FAMILY_ORDER:
            curve = degradation_curve(rows, controller, family)
            if not curve:
                continue
            bp, max_tested = break_point(curve, gate)
            families[family] = {
                "curve": [[sev, success] for sev, success in curve],
                "break_point": bp,
                "max_tested": max_tested,
            }
        summary_variants[label] = {
            "controller": controller,
            "nominal": nominal_success(rows, controller),
            "combined_max": combined_success(rows, controller),
            "families": families,
        }
    return {"card": card, "gate": gate, "variants": summary_variants}


def _format_break(bp: float | None, max_tested: float | None) -> str:
    if max_tested is None:
        return "n/a"
    if bp is None:
        return f"holds ≤{max_tested:g} (max tested)"
    return f"breaks at {bp:g}"


def plot_robustness_card(
    summary: dict[str, object],
    out_path: str | Path,
    episodes_per_cell: int | None = None,
) -> Path:
    """Render one controller's card (per-family degradation curves) to PNG.

    One panel per disturbance family, one line per variant, a horizontal gate
    reference, and a break-point line in the caption per family × variant.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    gate = float(summary["gate"])  # type: ignore[arg-type]
    variants: dict[str, dict[str, object]] = summary["variants"]  # type: ignore[assignment]
    families = [
        f
        for f in FAMILY_ORDER
        if any(f in v["families"] for v in variants.values())  # type: ignore[index]
    ]
    if not families or not variants:
        raise ValueError(f"card {summary['card']!r} has no plottable variants/families")

    fig, axes = plt.subplots(
        1, len(families), figsize=(1 + 3.4 * len(families), 3.8), squeeze=False, sharey=True
    )
    break_notes: list[str] = []
    for ax, family in zip(axes[0], families):
        ax.axhline(gate, color="#777777", linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        for idx, (label, vdata) in enumerate(variants.items()):
            fam = vdata["families"].get(family)  # type: ignore[union-attr]
            if fam is None:
                continue
            xs = [float(p[0]) for p in fam["curve"]]
            ys = [float(p[1]) for p in fam["curve"]]
            ax.plot(
                xs,
                ys,
                color=_VARIANT_COLORS[idx % len(_VARIANT_COLORS)],
                linestyle=_VARIANT_LINESTYLES[idx % len(_VARIANT_LINESTYLES)],
                marker=_VARIANT_MARKERS[idx % len(_VARIANT_MARKERS)],
                markersize=5,
                linewidth=2,
                label=label,
                zorder=2,
            )
            break_notes.append(
                f"{family}/{label}: {_format_break(fam['break_point'], fam['max_tested'])}"
            )
        if family in _SIGNED_FAMILIES:
            ax.axvline(0.0, color="#bbbbbb", linewidth=0.8, zorder=1)
        ax.set_xlabel(_FAMILY_LABELS.get(family, family))
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, color="#e6e6e6", linewidth=0.6, zorder=0)

    # Legend in the first (wind) panel: curves start near 1.0 there, so the
    # lower-left corner is reliably empty; later panels may collapse to 0.
    axes[0][0].set_ylabel("success rate")
    axes[0][0].legend(loc="lower left", fontsize=8, framealpha=0.9)

    headline_bits = []
    for label, vdata in variants.items():
        nominal = vdata.get("nominal")
        combined = vdata.get("combined_max")
        bits = [label]
        if nominal is not None:
            bits.append(f"nominal {float(nominal):.0%}")  # type: ignore[arg-type]
        if combined is not None:
            bits.append(f"combined-max {float(combined):.0%}")  # type: ignore[arg-type]
        headline_bits.append(" ".join(bits))
    fig.suptitle(
        f"{str(summary['card']).upper()} — robustness card "
        f"(gate {gate:.0%}) · " + " | ".join(headline_bits),
        fontsize=10,
    )

    caption = "break-points (gate): " + "; ".join(break_notes)
    if "sensor_noise" in families:
        caption += ". σ=0 level includes spike-only cells (not the nominal condition)"
    caption += "\ngraduated fixed-seed matrix; identical conditions per cell"
    if episodes_per_cell:
        caption += f"; n = {episodes_per_cell} episodes/cell"
    # Below the axes; bbox_inches="tight" expands the canvas to include it, so
    # the caption never collides with the x-axis labels.
    fig.text(0.5, -0.08, caption, ha="center", fontsize=7, style="italic")

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


def write_card_summary(summary: dict[str, object], out_path: str | Path) -> Path:
    """Write one card's summary as JSON (the exportable robustness report)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("wrote %s", out)
    return out
