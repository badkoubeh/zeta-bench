"""Hydra entrypoint: build per-controller robustness cards from matrix CSVs.

For each configured card (one per controller), merges the graduated-matrix
CSVs, computes per-family degradation curves and break-point severities for
every variant present (e.g. nominal-trained vs robust-trained), and writes a
PNG card plus a JSON summary. Curve/break-point logic lives in
:mod:`robustness.cards`; this file only wires config to outputs.

CLI
---
::

    python experiments/robustness_card.py
    python experiments/robustness_card.py robustness_card.gate=0.90

Outputs
-------
- ``results/cards/{card}.png`` — degradation curves per disturbance family.
- ``results/cards/{card}.json`` — serializable curves + break-points.
"""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from robustness.cards import (
    build_card_summary,
    load_matrix_rows,
    plot_robustness_card,
    write_card_summary,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="robustness_card", version_base=None)
def main(cfg: DictConfig) -> None:
    """Merge matrix CSVs and emit one card (PNG + JSON) per controller."""
    card_cfg = cfg.robustness_card
    rows: list[dict[str, object]] = []
    for csv_path in card_cfg.csv_paths:
        rows.extend(load_matrix_rows(csv_path))
    if not rows:
        raise RuntimeError(
            "no matrix rows loaded — run experiments/evaluate_robustness.py first "
            f"(looked for: {list(card_cfg.csv_paths)})"
        )

    episodes_per_cell = int(max(int(str(r["n_episodes"])) for r in rows))
    out_dir = Path(card_cfg.out_dir)
    built = 0
    for card, variants in card_cfg.cards.items():
        summary = build_card_summary(rows, str(card), dict(variants), float(card_cfg.gate))
        if not summary["variants"]:
            logger.warning("card %s: no variants present in matrix rows — skipping", card)
            continue
        plot_robustness_card(
            summary, out_dir / f"{card}.png", episodes_per_cell=episodes_per_cell
        )
        write_card_summary(summary, out_dir / f"{card}.json")
        built += 1
    logger.info("built %d card(s) in %s", built, out_dir)


if __name__ == "__main__":
    main()
