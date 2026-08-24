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


# ============================================================
# Reference collision kinematics from Python DH kinematics
# ============================================================

def collision_kinematics_python(robot, q, htm):
    jg_dh, htm_dh = robot.jac_geo(
        q=q,
        axis="dh",
        htm=htm,
        mode="python"
    )

    htm_A_all = []
    jac_A_all = []

    for i, link in enumerate(robot.links):
        H_i = np.asarray(htm_dh[i], dtype=float)
        J_i = np.asarray(jg_dh[i], dtype=float)

        p_i = H_i[0:3, 3]
        Jv_i = J_i[0:3, :]
        Jw_i = J_i[3:6, :]

        htm_A_link = []
        jac_A_link = []

        for col_obj in link.col_objects:
            H_iA = np.asarray(col_obj[1], dtype=float)
            H_A = H_i @ H_iA
            p_A = H_A[0:3, 3]
            r = p_A - p_i

            J_A = np.zeros_like(J_i)
            J_A[0:3, :] = Jv_i - skew(r) @ Jw_i
            J_A[3:6, :] = Jw_i

            htm_A_link.append(H_A)
            jac_A_link.append(J_A)

        htm_A_all.append(htm_A_link)
        jac_A_all.append(jac_A_link)

    return htm_A_all, jac_A_all


def relative_frobenius_error(reference, estimate):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    norm_reference = np.linalg.norm(reference, ord="fro")
    err = np.linalg.norm(reference - estimate, ord="fro")

    if norm_reference < 1e-12:
        return err

    return err / norm_reference


# ============================================================
# Monte Carlo validation
# ============================================================

robot = ub.Robot.create_franka_emika_3()
rng = np.random.default_rng()

num_configs = 2000
dt = 1e-3
error_threshold_percent = 0.1

# Arbitrary base with rotation and translation.
htm_base = np.matrix(
    trn(0.31, -0.22, 0.47)
    @ rotz(0.73)
    @ roty(-0.41)
    @ rotx(0.29)
)

num_cases = 0
num_success = 0

h_errors_percent = []
ja_errors_percent = []
jadot_errors_percent = []

for _ in range(num_configs):
    q = rng.uniform(-np.pi, np.pi, size=7)
    qdot = rng.uniform(-np.pi, np.pi, size=7)

    htm_A_cpp, jac_A_cpp, jacdot_A_cpp = robot.compute_col_object_kinematics_second_order(
        q=q,
        qdot=qdot,
        htm=htm_base
    )

    htm_A_ref, jac_A_ref = collision_kinematics_python(
        robot=robot,
        q=q,
        htm=htm_base
    )

    _, jac_A_prev = collision_kinematics_python(
        robot=robot,
        q=q - qdot * dt,
        htm=htm_base
    )

    _, jac_A_next = collision_kinematics_python(
        robot=robot,
        q=q + qdot * dt,
        htm=htm_base
    )

    for i in range(len(robot.links)):
        for j in range(len(robot.links[i].col_objects)):
            num_cases += 1

            h_error = relative_frobenius_error(
                htm_A_ref[i][j],
                htm_A_cpp[i][j]
            )
            h_errors_percent.append(100.0 * h_error)

            ja_error = relative_frobenius_error(
                jac_A_ref[i][j],
                jac_A_cpp[i][j]
            )
            ja_errors_percent.append(100.0 * ja_error)

            jacdot_A_num = (
                np.asarray(jac_A_next[i][j], dtype=float)
                -
                np.asarray(jac_A_prev[i][j], dtype=float)
            ) / (2.0 * dt)

            jadot_error = relative_frobenius_error(
                jacdot_A_num,
                jacdot_A_cpp[i][j]
            )
            jadot_error_percent = 100.0 * jadot_error
            jadot_errors_percent.append(jadot_error_percent)

            if jadot_error_percent < error_threshold_percent:
                num_success += 1

success_rate = 100.0 * num_success / num_cases

print()
print("=" * 60)
print("Collision object second-order kinematics validation")
print("=" * 60)
print(f"Number of random configurations: {num_configs}")
print(f"Number of primitive tests: {num_cases}")
print(f"Threshold: {error_threshold_percent:.6f}%")
print(f"Successes: {num_success}/{num_cases}")
print(f"Success rate: {success_rate:.2f}%")
print()
print(f"Max H_A relative error: {np.max(h_errors_percent):.6e}%")
print(f"Max J_A relative error: {np.max(ja_errors_percent):.6e}%")
print()
print(f"Mean Jdot_A relative error: {np.mean(jadot_errors_percent):.6e}%")
print(f"Maximum Jdot_A relative error: {np.max(jadot_errors_percent):.6e}%")
print(f"Minimum Jdot_A relative error: {np.min(jadot_errors_percent):.6e}%")
