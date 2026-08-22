import numpy as np
import uaibot as ub


def compute_numerically_jgdot(robot, q, qdot, dt):
    jg_prev, _ = robot.jac_geo(q=q - qdot * dt)
    jg_next, _ = robot.jac_geo(q=q + qdot * dt)

    return (jg_next - jg_prev) / (2 * dt)


def relative_frobenius_error(A, B):
    """
    Relative Frobenius error:

        ||A - B||_F
    e = -----------
           ||A||_F

    A is considered the reference matrix.
    """

    norm_A = np.linalg.norm(A, ord="fro")

    if norm_A < 1e-12:
        return np.linalg.norm(A - B, ord="fro")

    return np.linalg.norm(A - B, ord="fro") / norm_A


robot = ub.Robot.create_franka_emika_3()

num_tests = 50000
dt = 1e-3

# Maximum acceptable relative error, in %
error_threshold_percent = 0.1

num_success = 0
errors_percent = []

for i in range(num_tests):

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

    # Numerical reference: central difference in Python
    jgdot_num = compute_numerically_jgdot(
        robot=robot,
        q=q,
        qdot=qdot,
        dt=dt
    )

    # Analytical implementation in C++
    jgdot_cpp, _, _ = robot.compute_jgdot(
        q=q,
        qdot=qdot
    )

    relative_error = relative_frobenius_error(
        jgdot_num,
        jgdot_cpp
    )

    error_percent = 100 * relative_error
    errors_percent.append(error_percent)

    if error_percent < error_threshold_percent:
        num_success += 1


success_rate = 100 * num_success / num_tests

print(f"Number of tests: {num_tests}")
print(f"Threshold: {error_threshold_percent:.6f}%")
print(f"Successes: {num_success}/{num_tests}")
print(f"Success rate: {success_rate:.2f}%")

print()
print("C++ analytical vs Python central difference")
print(f"Mean relative error: {np.mean(errors_percent):.6e}%")
print(f"Maximum relative error: {np.max(errors_percent):.6e}%")
print(f"Minimum relative error: {np.min(errors_percent):.6e}%")