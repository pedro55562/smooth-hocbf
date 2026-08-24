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
    known_obs: list
    unknown_obs: list
    all_obs: list
    htm_path: np.ndarray
    htm_target: np.matrix
    ball_tr: object
    H: np.matrix
    xi: np.matrix
    dt: float
    dt_num: float
    t_max: float
    simular_movimento: bool


def draw_pc(path, sim, color="white", radius=0.02):
    sl = []
    for htm in path:
        sl.append(htm[0:3, 3])
    pc = ub.PointCloud(size=radius, color=color, points=sl)
    sim.add(pc)


def carregar_htm(nome_arquivo):
    pasta_script = os.path.dirname(os.path.abspath(__file__))
    caminho_arquivo = os.path.join(pasta_script, nome_arquivo)

    htms = []
    matriz_atual = []

    with open(caminho_arquivo, "r") as f:
        for linha in f:
            linha = linha.strip()

            if linha == "":
                if matriz_atual:
                    htms.append(np.matrix(matriz_atual))
                    matriz_atual = []
            else:
                valores = [float(v) for v in linha.split()]
                matriz_atual.append(valores)

        if matriz_atual:
            htms.append(np.matrix(matriz_atual))

    return np.array(htms)


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

    for ax in [ax1, ax2]:
        _apply_plot_style(ax, show_grid)

    if labels_u is not None and n_u > 1:
        ax1.legend(loc="upper right", frameon=False)

    if labels_xi is not None and n_xi > 1:
        ax2.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    _save_plot(fig, file_name, dpi, show_plot)


def create_scenario():
    sim = ub.Simulation.create_sim_hill()

    robot_body = ub.Cylinder(
        htm=ub.Utils.trn([0, 0, 0]) * ub.Utils.roty(np.pi),
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
        list_of_objects=[robot_body, robot_rigid_3d, robot_frame],
        htm=ub.Utils.trn([0, 0, 0.1]) * ub.Utils.roty(np.pi),
    )
    sim.add([robot_UAV])
    robot_body_copy = robot_body.copy()

    material_metal = ub.MeshMaterial.create_rough_metal()
    material_wood = ub.MeshMaterial.create_wood()

    piso = ub.Box(
        htm=ub.Utils.trn([0, 0, -0.2]),
        width=7,
        depth=7,
        height=0.05,
        mesh_material=material_wood,
    )
    teto = ub.Box(
        htm=ub.Utils.trn([0, 0, 1.74]),
        width=7,
        depth=7,
        height=0.05,
        mesh_material=material_wood,
    )
    parede_frente = ub.Box(
        htm=ub.Utils.trn([0, 2, 0.8]),
        width=3,
        depth=0.1,
        height=1.9,
        mesh_material=material_wood,
    )
    parede_fundo = ub.Box(
        htm=ub.Utils.trn([0, 3.5, 0.8]),
        width=7,
        depth=0.1,
        height=1.9,
        mesh_material=material_wood,
    )
    parede_lateral = ub.Box(
        htm=ub.Utils.trn([-1.5, 2.75, 0.8]) * ub.Utils.rotz(np.pi / 2),
        width=1.5,
        depth=0.1,
        height=1.9,
        mesh_material=material_wood,
    )
    parede_sup = ub.Box(
        htm=ub.Utils.trn([1.3, 2.42, 1.52]) * ub.Utils.rotz(np.pi / 2),
        width=0.75,
        depth=0.1,
        height=0.95,
        mesh_material=material_metal,
    )
    parede_inf = ub.Box(
        htm=ub.Utils.trn([1.3, 2.42, -0.5]) * ub.Utils.rotz(np.pi / 2),
        width=0.75,
        depth=0.1,
        height=0.95,
        mesh_material=material_metal,
    )
    parede_sup_lat = ub.Box(
        htm=ub.Utils.trn([1.3, 3.16, 0.8]) * ub.Utils.rotz(np.pi / 2),
        width=0.74,
        depth=0.1,
        height=1.9,
        mesh_material=material_metal,
    )
    pilar = ub.Cylinder(
        htm=ub.Utils.trn([1.35, 1, 1]),
        height=2,
        radius=0.05,
        mesh_material=material_metal,
    )

    unknown_obs = [parede_sup, parede_sup_lat, pilar]
    known_obs = [parede_frente, piso, teto, parede_fundo, parede_lateral, parede_inf]
    all_obs = known_obs + unknown_obs
    sim.add(all_obs)

    htm_path = carregar_htm("caminho.txt")
    htm_target = np.matrix(htm_path[-1])
    frame_target = ub.Frame(htm=htm_target)
    sim.add([frame_target])
    draw_pc(path=htm_path, sim=sim, color="white", radius=0.02)

    ball_tr = ub.Ball(htm=np.identity(4), radius=0.02, color="cyan")
    sim.add([ball_tr])

    dt = 0.01
    dt_num = 0.085
    t_max = 40.0
    H = np.matrix(robot_UAV.htm)
    xi = np.matrix(np.zeros((6, 1)))
    simular_movimento = True

    return Scenario(
        sim=sim,
        robot_body=robot_body,
        robot_body_copy=robot_body_copy,
        robot_UAV=robot_UAV,
        known_obs=known_obs,
        unknown_obs=unknown_obs,
        all_obs=all_obs,
        htm_path=htm_path,
        htm_target=htm_target,
        ball_tr=ball_tr,
        H=H,
        xi=xi,
        dt=dt,
        dt_num=dt_num,
        t_max=t_max,
        simular_movimento=simular_movimento,
    )
