import os
import sys

import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt

from setup import *


OUTPUT_DIR = "/home/pedro/Projects/smooth-hocbf/manipulator/"
ANIMATION_NAME = "hocbf_qp_cbf01"
FAIL_ANIMATION_NAME = "hocbf_qp_cbf01_failed"
HISTORY_FILE = "hocbf_qp_cbf01_histories.npz"

np.set_printoptions(precision=6, suppress=True)



# new_cyl = ub.simobjects.Cylinder(
#     radius=1,
#     height=1,
#     color="blue",
#     htm=ub.Utils.trn([10, 0, 0])
# )
robot, sim, all_obs, q0, htm_tg, htm_base = setup_motion_planning_simulation(problem_index=10)
# all_obs = [new_cyl]
objss = []
for link in robot.links:
    for col_obj_data in link.col_objects:
        objss.append(col_obj_data[0])
sim.add(objss)
# sim.add(new_cyl)


# ============================================================
# Simulation parameters
# ============================================================

dt = 0.01
t = 0.0
tmax = 6.0

# Pole of the critically damped task dynamics.
lam = 1.4

# Damping for the pseudoinverse.
damping = 1e-3

# Smooth HOCBF parameters.
lambda_safe = 0.03
eta = .5
h_smooth = 0.03
eps_smooth = 0.04
dist_tol = 1e-6
dist_no_iter_max = 8000

n = len(robot.q)

# Initial state.
q = np.asarray(q0, dtype=float).reshape(n)
qdot = np.zeros(n)
robot.add_ani_frame(time=0.0, q=q)

old_dist_structs = [None for _ in all_obs]

hist_r = []
hist_q = []
hist_qdot = []
hist_qddot = []
hist_t = []


def compute_nominal_acceleration(q, qdot):
    r, Jr, Jrdot = robot.compute_jrdot(
        htm_tg=htm_tg,
        q=q,
        qdot=qdot,
    )

    r = np.asarray(r, dtype=float).reshape(6, 1)
    Jr = np.asarray(Jr, dtype=float)
    Jrdot = np.asarray(Jrdot, dtype=float)
    qdot_col = qdot.reshape(n, 1)

    rdot = Jr @ qdot_col
    rddot_des = -2.0 * lam * rdot - lam**2 * r
    rhs = rddot_des - Jrdot @ qdot_col

    Jr_pinv = np.asarray(ub.Utils.dp_inv(Jr, damping), dtype=float)
    qddot_nom = (Jr_pinv @ rhs).reshape(n)

    return qddot_nom, r


def append_hocbf_constraints(obj, old_dist_struct, q, qdot):
    dist_struct = robot.compute_dist(
        obj=obj,
        q=q,
        qdot=qdot,
        h=h_smooth,
        eps=eps_smooth,
        tol=dist_tol,
        no_iter_max=dist_no_iter_max,
        max_dist=np.inf,
        old_dist_struct=old_dist_struct,
        mode="c++",
    )

    qdot_col = qdot.reshape(n, 1)
    A_rows = []
    b_rows = []
    distances = []
    Bs = []

    for item in dist_struct:
        distance = float(item.distance)
        jac_distance = np.asarray(item.jac_distance, dtype=float).reshape(1, n)
        jac_distance_dot = np.asarray(item.jac_distance_dot, dtype=float).reshape(1, n)

        B = distance - lambda_safe
        Bdot = float((jac_distance @ qdot_col)[0, 0])

        b_i = float((
            -jac_distance_dot @ qdot_col
            -2.0 * eta * Bdot
            -eta**2 * B
        )[0, 0])

        A_rows.append(jac_distance)
        b_rows.append([b_i])
        distances.append(distance)
        Bs.append(B)

    if len(A_rows) == 0:
        A = np.zeros((0, n))
        b = np.zeros((0, 1))
    else:
        A = np.vstack(A_rows)
        b = np.asarray(b_rows, dtype=float)

    return dist_struct, A, b, distances, Bs


def save_animation(file_name):
    sim.save(
        address=OUTPUT_DIR,
        file_name=file_name,
    )


def abort_on_qp_failure(exc, iteration, t, q, qdot, qddot_nom, A, b, min_distance, min_B):
    print("\nHOCBF-QP failed.")
    print(f"Exception: {repr(exc)}")
    print(f"time: {t:.6f}")
    print(f"iteration: {iteration}")
    print("q:")
    print(q)
    print("qdot:")
    print(qdot)
    print("qddot_nom:")
    print(qddot_nom)
    print("A:")
    print(A)
    print("b:")
    print(b.reshape(-1))
    print(f"minimum distance: {min_distance:.12f}")
    print(f"minimum B: {min_B:.12f}")

    save_animation(FAIL_ANIMATION_NAME)
    save_histories_and_plots()
    sys.exit(1)


def save_histories_and_plots():
    hist_r_arr = np.asarray(hist_r)
    hist_q_arr = np.asarray(hist_q)
    hist_qdot_arr = np.asarray(hist_qdot)
    hist_qddot_arr = np.asarray(hist_qddot)
    hist_t_arr = np.asarray(hist_t)

    np.savez(
        os.path.join(OUTPUT_DIR, HISTORY_FILE),
        t=hist_t_arr,
        r=hist_r_arr,
        q=hist_q_arr,
        qdot=hist_qdot_arr,
        qddot=hist_qddot_arr,
    )

    if hist_t_arr.size == 0:
        return

    plot_specs = [
        (hist_r_arr, "Task error r(t)", "Task error", "hocbf_qp_cbf01_task_error.png", "r"),
        (hist_q_arr, "Joint configuration q(t)", "Joint position [rad]", "hocbf_qp_cbf01_q.png", "q"),
        (hist_qdot_arr, "Joint velocity qdot(t)", "Joint velocity [rad/s]", "hocbf_qp_cbf01_qdot.png", "qdot"),
        (hist_qddot_arr, "Joint acceleration qddot(t) = u(t)", "Joint acceleration [rad/s^2]", "hocbf_qp_cbf01_qddot.png", "qddot"),
    ]

    for data, title, ylabel, filename, prefix in plot_specs:
        plt.figure(figsize=(9, 5))
        for ind in range(data.shape[1]):
            plt.plot(hist_t_arr, data[:, ind], label=f"{prefix}_{ind + 1}")
        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
        plt.close()


# ============================================================
# Main control loop
# ============================================================

for iteration in range(round(tmax / dt)):
    qddot_nom, r = compute_nominal_acceleration(q, qdot)

    A_blocks = []
    b_blocks = []
    all_distances = []
    all_Bs = []

    for obs_index, obs in enumerate(all_obs):
        dist_struct, A_obs, b_obs, distances, Bs = append_hocbf_constraints(
            obs,
            old_dist_structs[obs_index],
            q,
            qdot,
        )
        old_dist_structs[obs_index] = dist_struct

        A_blocks.append(A_obs)
        b_blocks.append(b_obs)
        all_distances.extend(distances)
        all_Bs.extend(Bs)

    A = np.vstack(A_blocks) if len(A_blocks) > 0 else np.zeros((0, n))
    b = np.vstack(b_blocks) if len(b_blocks) > 0 else np.zeros((0, 1))

    H = np.eye(n)
    f = -qddot_nom.reshape(n, 1)

    min_distance = min(all_distances) if len(all_distances) > 0 else float("inf")
    min_B = min(all_Bs) if len(all_Bs) > 0 else float("inf")

    try:
        qddot = np.asarray(
            ub.Utils.solve_qp(H, f, A, b),
            dtype=float,
        ).reshape(n)
    except Exception as exc:
        abort_on_qp_failure(
            exc,
            iteration,
            t,
            q,
            qdot,
            qddot_nom,
            A,
            b,
            min_distance,
            min_B,
        )

    hist_r.append(r.reshape(6))
    hist_q.append(q.copy())
    hist_qdot.append(qdot.copy())
    hist_qddot.append(qddot.copy())
    hist_t.append(t)

    q_next = q + qdot * dt + 0.5 * qddot * dt**2
    qdot_next = qdot + qddot * dt

    robot.add_ani_frame(
        time=t + dt,
        q=q_next,
    )

    q = q_next
    qdot = qdot_next
    t += dt


save_animation(ANIMATION_NAME)
save_histories_and_plots()
