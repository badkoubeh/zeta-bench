"""Hydra entrypoint: run the robustness disturbance sweep.

CLI
---
    python experiments/evaluate_robustness.py \\
        checkpoint=results/sac_moderate_adversarial_42/

Outputs
-------
- ``results/robustness_matrix.csv`` — full disturbance-sweep table
- ``results/side_by_side.mp4`` — naive vs robust agent in combined disturbance
- wandb ``Table`` artefact logged automatically

Disturbance grid (sketch)
-------------------------
::

    for wind_mag, wind_dir in product(wind.magnitudes, wind.directions):
        for mass_offset in mass_offset_fraction:
            for noise_sigma, spike_p in product(noise.sigma, noise.spike_probability):
                evaluate(seeds × episodes_per_seed)
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """Load checkpoint, sweep disturbance grid, write CSV + MP4 + wandb table."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
