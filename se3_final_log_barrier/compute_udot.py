import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROL_DATA_FILE = os.path.join(SCRIPT_DIR, "control_data.npz")
U_DOT_DATA_FILE = os.path.join(SCRIPT_DIR, "u_dot_data.npz")
U_DOT_PLOT_FILE = os.path.join(SCRIPT_DIR, "u_dot_savgol.pdf")

POLYORDER = 2
DEFAULT_WINDOW_LENGTH = 10


def adjusted_window_length(num_samples):
    window_length = min(DEFAULT_WINDOW_LENGTH, num_samples)
    if window_length % 2 == 0:
        window_length -= 1

    if window_length <= POLYORDER:
        raise ValueError(
            f"At least {POLYORDER + 2} samples are required for "
            f"polyorder={POLYORDER}; received {num_samples}."
        )

    return window_length


with np.load(CONTROL_DATA_FILE) as data:
    u = np.asarray(data["u"]).reshape(-1, 6)
    time = np.asarray(data["time"])
    dt = float(data["dt"])

if len(u) != len(time):
    raise ValueError(
        f"u and time must have the same number of samples; "
        f"received {len(u)} and {len(time)}."
    )

window_length = adjusted_window_length(len(u))
u_dot = savgol_filter(
    u,
    window_length=window_length,
    polyorder=POLYORDER,
    deriv=1,
    delta=dt,
    axis=0,
    mode="interp",
)

np.savez(U_DOT_DATA_FILE, u_dot=u_dot, time=time, dt=dt)

labels = [
    r"$\dot{u}_{v_x}$",
    r"$\dot{u}_{v_y}$",
    r"$\dot{u}_{v_z}$",
    r"$\dot{u}_{\omega_x}$",
    r"$\dot{u}_{\omega_y}$",
    r"$\dot{u}_{\omega_z}$",
]

fig, axes = plt.subplots(3, 2, figsize=(8, 8), sharex=True)
for component, ax in enumerate(axes.flat):
    ax.plot(time, u_dot[:, component], linewidth=1.2)
    ax.set_ylabel(labels[component])
    ax.grid(True, alpha=0.3)

for ax in axes[-1, :]:
    ax.set_xlabel("Time (s)")

fig.tight_layout()
fig.savefig(U_DOT_PLOT_FILE, bbox_inches="tight")
plt.close(fig)

print(f"Savitzky-Golay window length: {window_length}")
print(f"Saved derivative data to: {U_DOT_DATA_FILE}")
print(f"Saved plot to: {U_DOT_PLOT_FILE}")
