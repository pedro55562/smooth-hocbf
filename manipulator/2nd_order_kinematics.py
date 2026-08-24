import numpy as np
import uaibot as ub


def compute_numerically_jgdot(robot, q, qdot, dt):
    """
    Compute Jgdot numerically for all DH frames using central difference.
    """

    jg_prev, _ = robot.jac_geo(
        q=q - qdot * dt,
        axis="dh",
        mode="python"
    )

    jg_next, _ = robot.jac_geo(
        q=q + qdot * dt,
        axis="dh",
        mode="python"
    )

    jgdot_num = []

    for jg_prev_i, jg_next_i in zip(jg_prev, jg_next):
        jgdot_num.append(
            (
                np.asarray(jg_next_i, dtype=float)
                - np.asarray(jg_prev_i, dtype=float)
            ) / (2 * dt)
        )

    return jgdot_num


def relative_frobenius_error(A, B):
    """
    Relative Frobenius error:

        ||A - B||_F
    e = -----------
           ||A||_F

    A is considered the reference matrix.
    """

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    norm_A = np.linalg.norm(A, ord="fro")

    if norm_A < 1e-12:
        return np.linalg.norm(A - B, ord="fro")

    return np.linalg.norm(A - B, ord="fro") / norm_A


robot = ub.Robot.create_franka_emika_3()

num_tests = 100
dt = 1e-3

# Maximum acceptable relative error, in %
error_threshold_percent = 0.1


# ============================================================
# Statistics
# ============================================================

num_success_jg = 0
num_success_jgdot = 0

errors_jg_percent = []
errors_jgdot_percent = []


# ============================================================
# Monte Carlo
# ============================================================

for test_idx in range(num_tests):

    q = np.random.uniform(
        -np.pi,
        np.pi,
        size=7
    )

    qdot = np.random.uniform(
        -np.pi,
        np.pi,
        size=7
    )


    # ========================================================
    # Python reference Jg for all DH frames
    # ========================================================

    jg_python, _ = robot.jac_geo(
        q=q,
        axis="dh",
        mode="python"
    )


    # ========================================================
    # Numerical Jgdot for all DH frames
    # ========================================================

    jgdot_num = compute_numerically_jgdot(
        robot=robot,
        q=q,
        qdot=qdot,
        dt=dt
    )


    # ========================================================
    # C++ analytical implementation
    #
    # For axis="dh":
    #
    # fkm_cpp[i]   -> 4x4
    # jg_cpp[i]    -> 6xn
    # jgdot_cpp[i] -> 6xn
    # ========================================================

    _, jg_cpp, jgdot_cpp = robot.compute_jgdot(
        q=q,
        qdot=qdot,
        axis="dh"
    )


    # ========================================================
    # Validate every DH frame
    # ========================================================

    test_success_jg = True
    test_success_jgdot = True

    for frame_idx in range(len(jg_cpp)):

        # ----------------------------------------------------
        # Jg
        # ----------------------------------------------------

        error_jg = relative_frobenius_error(
            jg_python[frame_idx],
            jg_cpp[frame_idx]
        )

        error_jg_percent = 100 * error_jg

        errors_jg_percent.append(
            error_jg_percent
        )

        if error_jg_percent >= error_threshold_percent:
            test_success_jg = False


        # ----------------------------------------------------
        # Jgdot
        # ----------------------------------------------------

        error_jgdot = relative_frobenius_error(
            jgdot_num[frame_idx],
            jgdot_cpp[frame_idx]
        )

        error_jgdot_percent = 100 * error_jgdot

        errors_jgdot_percent.append(
            error_jgdot_percent
        )

        if error_jgdot_percent >= error_threshold_percent:
            test_success_jgdot = False


    # A test is considered successful only if ALL DH frames
    # satisfy the threshold.

    if test_success_jg:
        num_success_jg += 1

    if test_success_jgdot:
        num_success_jgdot += 1


# ============================================================
# Results
# ============================================================

success_rate_jg = (
    100 * num_success_jg / num_tests
)

success_rate_jgdot = (
    100 * num_success_jgdot / num_tests
)


print()
print("=" * 60)
print("Jg validation - all DH frames")
print("=" * 60)

print(f"Number of tests: {num_tests}")
print(f"Threshold: {error_threshold_percent:.6f}%")
print(
    f"Successes: "
    f"{num_success_jg}/{num_tests}"
)
print(
    f"Success rate: "
    f"{success_rate_jg:.2f}%"
)

print()
print("C++ analytical Jg vs Python Jg")
print(
    f"Mean relative error: "
    f"{np.mean(errors_jg_percent):.6e}%"
)
print(
    f"Maximum relative error: "
    f"{np.max(errors_jg_percent):.6e}%"
)
print(
    f"Minimum relative error: "
    f"{np.min(errors_jg_percent):.6e}%"
)


print()
print("=" * 60)
print("Jgdot validation - all DH frames")
print("=" * 60)

print(f"Number of tests: {num_tests}")
print(f"Threshold: {error_threshold_percent:.6f}%")
print(
    f"Successes: "
    f"{num_success_jgdot}/{num_tests}"
)
print(
    f"Success rate: "
    f"{success_rate_jgdot:.2f}%"
)

print()
print("C++ analytical Jgdot vs Python central difference")
print(
    f"Mean relative error: "
    f"{np.mean(errors_jgdot_percent):.6e}%"
)
print(
    f"Maximum relative error: "
    f"{np.max(errors_jgdot_percent):.6e}%"
)
print(
    f"Minimum relative error: "
    f"{np.min(errors_jgdot_percent):.6e}%"
)