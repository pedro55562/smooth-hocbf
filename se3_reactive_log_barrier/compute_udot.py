import os

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONTROL_DATA_FILE = os.path.join(SCRIPT_DIR, "control_data.npz")
U_DOT_DATA_FILE = os.path.join(SCRIPT_DIR, "u_dot_data.npz")
U_DOT_PLOT_FILE = os.path.join(SCRIPT_DIR, "u_dot_central.pdf")


# ============================
# Load data
# ============================

with np.load(CONTROL_DATA_FILE) as data:
    u = np.asarray(data["u"], dtype=float).reshape(-1, 6)
    time = np.asarray(data["time"], dtype=float)
    dt = float(data["dt"])


if len(u) != len(time):
    raise ValueError(
        f"u and time must have the same number of samples; "
        f"received {len(u)} and {len(time)}."
    )

if len(u) < 3:
    raise ValueError("At least 3 samples are required.")


# ============================
# Numerical derivative
# Central difference
# ============================

u_dot = np.gradient(
    u,
    time,
    axis=0,
    edge_order=2,
)


# ============================
# Save derivative
# ============================

np.savez(
    U_DOT_DATA_FILE,
    u_dot=u_dot,
    u=u,
    time=time,
    dt=dt,
)


# ============================
# Plot
# ============================

labels = [
    r"$\dot{u}_{v_x}$",
    r"$\dot{u}_{v_y}$",
    r"$\dot{u}_{v_z}$",
    r"$\dot{u}_{\omega_x}$",
    r"$\dot{u}_{\omega_y}$",
    r"$\dot{u}_{\omega_z}$",
]

fig, axes = plt.subplots(
    3,
    2,
    figsize=(8, 8),
    sharex=True,
)

for component, ax in enumerate(axes.flat):

    ax.plot(
        time,
        u_dot[:, component],
        linewidth=1.2,
    )

    ax.set_ylabel(labels[component])
    ax.grid(True, alpha=0.3)


for ax in axes[-1, :]:
    ax.set_xlabel("Time (s)")


fig.tight_layout()

fig.savefig(
    U_DOT_PLOT_FILE,
    bbox_inches="tight",
)

plt.close(fig)


print("Derivative method: second-order central finite difference")
print(f"Saved derivative data to: {U_DOT_DATA_FILE}")
print(f"Saved plot to: {U_DOT_PLOT_FILE}")