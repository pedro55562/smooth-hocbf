import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import uaibot as ub


@dataclass
class Scenario:
    sim: object
    robot_body: object
    robot_body_copy: object
    robot_UAV: object
    all_obs: list
    H_d: np.matrix
    H: np.matrix
    xi: np.matrix
    dt: float
    t_max: float
    simular_movimento: bool


def _process_series(data_list, t):
    processed = []
    for d in data_list:
        d = np.asarray(d).squeeze()

        if d.ndim == 0:
            d = d.reshape(1)
        elif d.ndim != 1:
            raise ValueError("Each element must be scalar or 1D.")

        processed.append(d)

    n = processed[0].shape[0]
    for d in processed:
        if d.shape[0] != n:
            raise ValueError("Dimension mismatch in data_list.")

    data = np.vstack(processed)
    if len(t) != data.shape[0]:
        raise ValueError("Time vector size mismatch.")

    return data, n


def _apply_plot_style(ax, show_grid=True):
    if show_grid:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        length=4,
        width=0.8,
        colors="black",
    )


def _configure_plot_style(linewidth):
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": linewidth,
        "axes.linewidth": 1.0,
    })


def _save_plot(fig, file_name, dpi, show_plot):
    base_dir = os.path.dirname(__file__)
    save_path = os.path.join(base_dir, file_name)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close(fig)
    print(f"Plot saved at: {save_path}")


def plot_dist_min(
    data_list,
    t,
    file_name,
    labels=None,
    xlabel="Time (s)",
    ylabel="Value",
    title=None,
    figsize=(6.0, 3.6),
    linewidth=1.2,
    dpi=300,
    show_grid=True,
    show_plot=False,
):
    if len(data_list) == 0:
        return

    data, n = _process_series(data_list, t)
    _configure_plot_style(linewidth)
    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n):
        label = labels[i] if labels is not None else f"$x_{i + 1}$"
        ax.plot(t, data[:, i], label=label)

    safety_distance = 0.01
    ax.axhline(
        y=safety_distance,
        linestyle="--",
        linewidth=1.2,
        color="red",
    )
    ax.text(
        t[-1],
        safety_distance,
        " safety limit",
        color="red",
        fontsize=9,
        va="bottom",
        ha="right",
    )

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _apply_plot_style(ax, show_grid)

    if labels is not None and n > 1:
        ax.legend(frameon=False)

    fig.tight_layout()
    _save_plot(fig, file_name, dpi, show_plot)


def plot_u_xi(
    u_list,
    xi_list,
    t,
    file_name,
    labels_u=None,
    labels_xi=None,
    xlabel="Time (s)",
    ylabel_u=r"$u$",
    ylabel_xi=r"$\xi$",
    title_u="Control Input",
    title_xi="System Twist",
    figsize=(6.0, 5.5),
    linewidth=1.2,
    dpi=300,
    show_grid=True,
    show_plot=False,
):
    if len(u_list) == 0 or len(xi_list) == 0:
        return

    data_u, n_u = _process_series(u_list, t)
    data_xi, n_xi = _process_series(xi_list, t)
    _configure_plot_style(linewidth)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    for i in range(n_u):
        label = labels_u[i] if labels_u is not None else f"$u_{i + 1}$"
        ax1.plot(t, data_u[:, i], label=label)

    ax1.set_ylabel(ylabel_u)
    if title_u:
        ax1.set_title(title_u)

    for i in range(n_xi):
        label = labels_xi[i] if labels_xi is not None else f"$x_{i + 1}$"
        ax2.plot(t, data_xi[:, i], label=label)

    ax2.set_ylabel(ylabel_xi)
    ax2.set_xlabel(xlabel)
    if title_xi:
        ax2.set_title(title_xi)

    for ax in (ax1, ax2):
        _apply_plot_style(ax, show_grid)

    if labels_u is not None and n_u > 1:
        ax1.legend(loc="upper right", frameon=False)
    if labels_xi is not None and n_xi > 1:
        ax2.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    _save_plot(fig, file_name, dpi, show_plot)


def plot_pose_error(
    error_list,
    t,
    file_name,
    tolerance=0.025,
    show_plot=False,
):
    if len(error_list) == 0:
        return

    _configure_plot_style(1.2)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(t, error_list)
    ax.axhline(
        y=tolerance,
        linestyle="--",
        linewidth=1.2,
        color="red",
        label="Tolerance",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$d(H, H_d)$")
    ax.set_title("Pose Error")
    ax.legend(frameon=False)
    _apply_plot_style(ax)
    fig.tight_layout()
    _save_plot(fig, file_name, 300, show_plot)




def create_scenario():
    sim = ub.Simulation.create_sim_hill()

    initial_htm = (
        ub.Utils.trn([-3.0, -1.2, 0.8])
        * ub.Utils.roty(np.pi)
    )

    H_d = (
        ub.Utils.trn([3.0, 1.2, 0.8])
        * ub.Utils.rotz(np.pi / 3)
        * ub.Utils.roty(np.pi)
    )

    # ============================
    # Robot
    # ============================

    robot_body = ub.Cylinder(
        htm=ub.Utils.roty(np.pi),
        name="robot_body",
        radius=0.3,
        height=0.17,
        color="cyan",
        opacity=0.55,
    )

    robot_3d_model = ub.Model3D(
        url="https://cdn.jsdelivr.net/gh/pedro55562/SE3_CBF_ASSETS@main/TEMA12_DRONA6.obj",
        scale=0.0009,
        mesh_material=ub.MeshMaterial.create_rough_metal(),
    )

    robot_frame = ub.Frame(size=0.10)

    robot_rigid_3d = ub.RigidObject(
        list_model_3d=[robot_3d_model],
        htm=ub.Utils.trn([0, 0, -0.05]) * ub.Utils.roty(np.pi),
    )

    robot_UAV = ub.Group(
        list_of_objects=[
            robot_body,
            robot_rigid_3d,
            robot_frame,
        ],
        htm=initial_htm,
    )

    sim.add([robot_UAV])

    robot_body_copy = robot_body.copy()

    # ============================
    # Environment
    # ============================

    material_metal = ub.MeshMaterial.create_rough_metal()

    ground = ub.Box(
        htm=ub.Utils.trn([0, 0, -0.05]),
        width=8,
        depth=5,
        height=0.05,
        color="gray",
        opacity=0.35,
    )

    # First obstacle: forces deviation upward in y
    obstacle_1 = ub.Cylinder(
        htm=ub.Utils.trn([-1.8, -0.95, 0.8]),
        height=1.3,
        radius=0.40,
        mesh_material=material_metal,
    )

    # Second obstacle: opposite side
    obstacle_2 = ub.Cylinder(
        htm=ub.Utils.trn([-0.55, 0.75, 0.8]),
        height=1.3,
        radius=0.42,
        mesh_material=material_metal,
    )

    # Third obstacle: forces another change of direction
    obstacle_3 = ub.Cylinder(
        htm=ub.Utils.trn([0.75, -0.60, 0.8]),
        height=1.3,
        radius=0.40,
        mesh_material=material_metal,
    )

    # Final obstacle near the target direction
    obstacle_4 = ub.Cylinder(
        htm=ub.Utils.trn([1.85, 0.95, 0.8]),
        height=1.3,
        radius=0.38,
        mesh_material=material_metal,
    )

    all_obs = [
        obstacle_1,
        obstacle_2,
        obstacle_3,
        obstacle_4,
    ]

    sim.add([ground] + all_obs)

    # ============================
    # Target
    # ============================

    frame_target = ub.Frame(
        htm=H_d,
        size=0.18,
    )

    sim.add([frame_target])

    # ============================
    # Simulation parameters
    # ============================

    dt = 0.05
    t_max = 40

    H = np.matrix(initial_htm)
    xi = np.matrix(np.zeros((6, 1)))

    simular_movimento = True

    return Scenario(
        sim=sim,
        robot_body=robot_body,
        robot_body_copy=robot_body_copy,
        robot_UAV=robot_UAV,
        all_obs=all_obs,
        H_d=np.matrix(H_d),
        H=H,
        xi=xi,
        dt=dt,
        t_max=t_max,
        simular_movimento=simular_movimento,
    )
