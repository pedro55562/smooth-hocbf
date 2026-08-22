from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _prepare_vectors(values) -> np.ndarray:
    if not values:
        return np.empty((0, 6))
    return np.vstack([np.asarray(value, dtype=float).reshape(1, -1) for value in values])


def plot_minimum_distance(
    distances,
    time,
    safety_distance: float,
    output_path: str | Path,
) -> None:
    if not distances:
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(time, distances)
    ax.axhline(safety_distance, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$d_{min}$")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_twist_comparison(
    nominal_twists,
    safe_twists,
    time,
    output_path: str | Path,
) -> None:
    if not nominal_twists or not safe_twists:
        return

    nominal = _prepare_vectors(nominal_twists)
    safe = _prepare_vectors(safe_twists)

    labels = [
        r"$v_x$", r"$v_y$", r"$v_z$",
        r"$\omega_x$", r"$\omega_y$", r"$\omega_z$",
    ]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)

    for i in range(6):
        axes[0].plot(time, nominal[:, i], label=labels[i])
        axes[1].plot(time, safe[:, i], label=labels[i])

    axes[0].set_ylabel(r"$\xi_d$")
    axes[0].set_title("Nominal twist")
    axes[1].set_ylabel(r"$\xi$")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("CBF-filtered twist")

    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(frameon=False, ncol=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
