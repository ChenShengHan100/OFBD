import argparse
import json
import math
import os
import sys
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot eig-density curve from approximate Hessian eigenvalues."
    )
    parser.add_argument(
        "--spectrum-json",
        default="./log_sup/standalone_ce_hessian_spectrum_all_dense.json",
    )
    parser.add_argument(
        "--output",
        default="./log_sup/standalone_ce_hessian_density_all_1e-2_1e2.png",
    )
    parser.add_argument("--x-min", default=1e-2, type=float)
    parser.add_argument("--x-max", default=1e2, type=float)
    parser.add_argument("--y-min", default=1e-2, type=float)
    parser.add_argument("--y-max", default=1e2, type=float)
    parser.add_argument("--num-points", default=400, type=int)
    parser.add_argument(
        "--bandwidth",
        default=0.12,
        type=float,
        help="Gaussian KDE bandwidth in log10-eigenvalue space.",
    )
    return parser.parse_args()


def logspace_kde(eigenvalues: List[float], xs: np.ndarray, bandwidth: float) -> np.ndarray:
    vals = np.asarray([v for v in eigenvalues if v > 0], dtype=np.float64)
    if vals.size == 0:
        raise ValueError("No positive eigenvalues available for density plot.")

    log_vals = np.log10(vals)
    log_xs = np.log10(xs)

    density_log = np.zeros_like(log_xs)
    coeff = 1.0 / (vals.size * bandwidth * math.sqrt(2.0 * math.pi))
    for lv in log_vals:
        z = (log_xs - lv) / bandwidth
        density_log += np.exp(-0.5 * z * z)
    density_log *= coeff

    # Convert density over log10(lambda) to density over lambda.
    density_x = density_log / (xs * math.log(10.0))
    return density_x


def main() -> None:
    args = parse_args()
    with open(args.spectrum_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    eigenvalues = payload["topk_eigenvalues"]
    xs = np.logspace(np.log10(args.x_min), np.log10(args.x_max), args.num_points)
    ys = logspace_kde(eigenvalues, xs, args.bandwidth)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, color="#1f77b4", linewidth=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(args.x_min, args.x_max)
    ax.set_ylim(args.y_min, args.y_max)
    ax.set_xlabel("eig")
    ax.set_ylabel("density")
    ax.set_title("Approximate Hessian Eig-Density")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    summary = {
        "spectrum_json": args.spectrum_json,
        "output": args.output,
        "x_range": [args.x_min, args.x_max],
        "y_range": [args.y_min, args.y_max],
        "bandwidth": args.bandwidth,
        "num_eigenvalues": len(eigenvalues),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
