"""Generate a reproducible, educational network-flow dataset.

The generated records imitate benign and suspicious traffic patterns. They are
not copied from a production network and contain no personal information.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "duration_seconds",
    "source_bytes",
    "destination_bytes",
    "packet_count",
    "failed_logins",
    "same_service_rate",
    "syn_flag_count",
    "unique_destination_ports",
]


def generate_dataset(rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Return a labelled synthetic network-flow dataset."""
    if rows < 100:
        raise ValueError("rows must be at least 100")

    rng = np.random.default_rng(seed)
    attacks = rng.random(rows) < 0.28

    duration = rng.gamma(shape=2.0, scale=3.0, size=rows)
    source_bytes = rng.lognormal(mean=7.1, sigma=0.8, size=rows)
    destination_bytes = rng.lognormal(mean=7.0, sigma=0.9, size=rows)
    packet_count = rng.poisson(lam=18, size=rows) + 1
    failed_logins = rng.poisson(lam=0.12, size=rows)
    same_service_rate = rng.beta(a=7.0, b=2.0, size=rows)
    syn_flag_count = rng.poisson(lam=1.0, size=rows)
    unique_ports = rng.poisson(lam=2.0, size=rows) + 1

    attack_idx = np.where(attacks)[0]
    duration[attack_idx] *= rng.uniform(0.05, 0.5, size=len(attack_idx))
    source_bytes[attack_idx] *= rng.uniform(0.2, 0.9, size=len(attack_idx))
    destination_bytes[attack_idx] *= rng.uniform(0.05, 0.7, size=len(attack_idx))
    packet_count[attack_idx] += rng.poisson(lam=35, size=len(attack_idx))
    failed_logins[attack_idx] += rng.poisson(lam=3.0, size=len(attack_idx))
    same_service_rate[attack_idx] = rng.beta(a=2.0, b=5.0, size=len(attack_idx))
    syn_flag_count[attack_idx] += rng.poisson(lam=10.0, size=len(attack_idx))
    unique_ports[attack_idx] += rng.poisson(lam=14.0, size=len(attack_idx))

    frame = pd.DataFrame(
        {
            "duration_seconds": duration.round(4),
            "source_bytes": source_bytes.round(2),
            "destination_bytes": destination_bytes.round(2),
            "packet_count": packet_count,
            "failed_logins": failed_logins,
            "same_service_rate": same_service_rate.round(4),
            "syn_flag_count": syn_flag_count,
            "unique_destination_ports": unique_ports,
            "label": np.where(attacks, "attack", "normal"),
        }
    )
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample IDS flow data")
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/network_flows.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset(rows=args.rows, seed=args.seed)
    dataset.to_csv(args.output, index=False)
    print(f"Created {len(dataset)} records at {args.output}")


if __name__ == "__main__":
    main()
