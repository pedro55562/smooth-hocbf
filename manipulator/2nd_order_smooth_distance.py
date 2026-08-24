import numpy as np
import uaibot as ub


# ============================================================
# Basic utilities
# ============================================================

def skew(v):
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0]
    ])


def rotx(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    H = np.eye(4)
    H[0:3, 0:3] = np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ])
    return H


def roty(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    H = np.eye(4)
    H[0:3, 0:3] = np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ])
    return H


def rotz(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    H = np.eye(4)
    H[0:3, 0:3] = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ])
    return H


def trn(x, y, z):
    H = np.eye(4)
    H[0:3, 3] = np.array([x, y, z], dtype=float)
    return H


def relative_frobenius_error(reference, estimate):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    norm_reference = np.linalg.norm(reference, ord="fro")
    err = np.linalg.norm(reference - estimate, ord="fro")

    if norm_reference < 1e-12:
        return err

    return err / norm_reference


def sample_configuration(robot, rng):
    limits = np.asarray(robot.joint_limit, dtype=float)
    margin = 0.08 * (limits[:, 1] - limits[:, 0])
    q_min = limits[:, 0] + margin
    q_max = limits[:, 1] - margin
    return rng.uniform(q_min, q_max)


def compute_distance_struct(robot, obj, q, qdot, htm_base, h, eps, tol, no_iter_max):
    return robot.compute_dist(
        obj=obj,
        q=q,
        htm=htm_base,
        qdot=qdot,
        h=h,
        eps=eps,
        tol=tol,
        no_iter_max=no_iter_max,
        max_dist=np.inf,
        mode="c++"
    )


def central_difference_jac_distance(robot, obj, q, qdot, htm_base, h, eps, tol, no_iter_max, dt):
    ds_prev = compute_distance_struct(robot, obj, q - qdot * dt, qdot, htm_base, h, eps, tol, no_iter_max)
    ds_next = compute_distance_struct(robot, obj, q + qdot * dt, qdot, htm_base, h, eps, tol, no_iter_max)

    return (
        np.asarray(ds_next.jac_dist_mat, dtype=float)
        - np.asarray(ds_prev.jac_dist_mat, dtype=float)
    ) / (2.0 * dt)


def d_xi_lambda_for_item(item, H_A, J_A, qdot):
    H_A = np.asarray(H_A, dtype=float)
    qdot = np.asarray(qdot, dtype=float).reshape((-1, 1))

    s_A = H_A[0:3, 3].reshape((3, 1))
    a_star = np.asarray(item.point_link, dtype=float).reshape((3, 1))
    b_star = np.asarray(item.point_object, dtype=float).reshape((3, 1))
    denom = float(item.distance) + 1e-6

    D_xi_lambda = np.zeros((1, 6))
    D_xi_lambda[:, 0:3] = ((a_star - b_star).T / denom)
    D_xi_lambda[:, 3:6] = ((skew(b_star - s_A) @ (a_star - b_star)).T / denom)

    return D_xi_lambda


# ============================================================
# Monte Carlo validation
# ============================================================

robot = ub.Robot.create_franka_emika_3()
rng = np.random.default_rng()

target_num_configs = 1000
max_attempts = 10000
dt = 1e-3
dt_stability = 7e-4
error_threshold_percent = 0.1
finite_difference_stability_percent = 0.05
min_smooth_distance = 0.15

h = 0.08
eps = 0.02
tol = 1e-7
no_iter_max = 8000

# Arbitrary robot base with rotation and translation.
htm_base = np.matrix(
    trn(0.31, -0.22, 0.47)
    @ rotz(0.73)
    @ roty(-0.41)
    @ rotx(0.29)
)

# External object B, also with arbitrary pose. It is kept away from collision so
# the finite-difference check stays in the smooth, well-conditioned region.
obj = ub.Box(
    htm=np.matrix(
        trn(1.15, -0.85, 1.15)
        @ rotz(-0.35)
        @ roty(0.22)
        @ rotx(0.18)
    ),
    width=0.18,
    depth=0.24,
    height=0.16,
    color="cyan"
)

num_success = 0
num_attempts = 0
num_rejected_distance = 0
num_rejected_stability = 0
num_identity_tests = 0

jdot_errors_percent = []
identity_errors_percent = []

while len(jdot_errors_percent) < target_num_configs and num_attempts < max_attempts:
    num_attempts += 1
    q = sample_configuration(robot, rng)
    qdot = rng.uniform(-0.7, 0.7, size=len(robot.links))

    ds = compute_distance_struct(robot, obj, q, qdot, htm_base, h, eps, tol, no_iter_max)

    if np.min(np.asarray(ds.dist_vect, dtype=float)) < min_smooth_distance:
        num_rejected_distance += 1
        continue

    jac_distance_dot_num = central_difference_jac_distance(
        robot, obj, q, qdot, htm_base, h, eps, tol, no_iter_max, dt
    )
    jac_distance_dot_num_stability = central_difference_jac_distance(
        robot, obj, q, qdot, htm_base, h, eps, tol, no_iter_max, dt_stability
    )

    stability_error = 100.0 * relative_frobenius_error(
        jac_distance_dot_num,
        jac_distance_dot_num_stability
    )
    if stability_error > finite_difference_stability_percent:
        num_rejected_stability += 1
        continue

    jac_distance_dot_ana = np.asarray(ds.jac_dist_dot_mat, dtype=float)
    jdot_error = relative_frobenius_error(jac_distance_dot_num, jac_distance_dot_ana)
    jdot_error_percent = 100.0 * jdot_error
    jdot_errors_percent.append(jdot_error_percent)

    if jdot_error_percent < error_threshold_percent:
        num_success += 1

    H_A_all, J_A_all, _ = robot.compute_col_object_kinematics_second_order(
        q=q,
        qdot=qdot,
        htm=htm_base
    )

    jac_from_lambda_rows = []
    for k in range(ds.no_items):
        item = ds[k]
        i = item.link_number
        j = item.link_col_obj_number
        D_xi_lambda = d_xi_lambda_for_item(item, H_A_all[i][j], J_A_all[i][j], qdot)
        jac_from_lambda = D_xi_lambda @ np.asarray(J_A_all[i][j], dtype=float)
        jac_from_lambda_rows.append(jac_from_lambda)

        identity_error = relative_frobenius_error(
            np.asarray(item.jac_distance, dtype=float),
            jac_from_lambda
        )
        identity_errors_percent.append(100.0 * identity_error)
        num_identity_tests += 1

    jac_from_lambda_mat = np.vstack(jac_from_lambda_rows)
    identity_matrix_error = relative_frobenius_error(
        np.asarray(ds.jac_dist_mat, dtype=float),
        jac_from_lambda_mat
    )
    identity_errors_percent.append(100.0 * identity_matrix_error)

if len(jdot_errors_percent) < target_num_configs:
    raise RuntimeError("Could not collect enough stable smooth-distance validation samples.")

success_rate = 100.0 * num_success / len(jdot_errors_percent)

print()
print("=" * 68)
print("Smooth distance jac_distance_dot validation")
print("=" * 68)
print(f"Accepted random configurations: {len(jdot_errors_percent)}")
print(f"Sampled random configurations: {num_attempts}")
print(f"Rejected by minimum distance: {num_rejected_distance}")
print(f"Rejected by finite-difference stability: {num_rejected_stability}")
print(f"Number of D_xi_lambda @ J_A primitive tests: {num_identity_tests}")
print(f"Threshold: {error_threshold_percent:.6f}%")
print(f"Successes: {num_success}/{len(jdot_errors_percent)}")
print(f"Success rate: {success_rate:.2f}%")
print()
print("jac_distance_dot relative Frobenius error")
print(f"Mean: {np.mean(jdot_errors_percent):.6e}%")
print(f"Maximum: {np.max(jdot_errors_percent):.6e}%")
print(f"Minimum: {np.min(jdot_errors_percent):.6e}%")
print()
print("D_xi_lambda @ J_A versus current jac_distance")
print(f"Mean: {np.mean(identity_errors_percent):.6e}%")
print(f"Maximum: {np.max(identity_errors_percent):.6e}%")
print(f"Minimum: {np.min(identity_errors_percent):.6e}%")
