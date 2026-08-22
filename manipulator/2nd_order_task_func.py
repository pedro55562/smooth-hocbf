import numpy as np
import uaibot as ub


# ============================================================
# Basic utilities
# ============================================================

def skew(v):
    """
    Skew-symmetric matrix S(v) such that:

        S(v) @ w = v x w
    """

    x, y, z = np.asarray(v, dtype=float).reshape(3)

    return np.array([
        [0.0, -z,   y],
        [z,    0.0, -x],
        [-y,   x,   0.0]
    ])


def random_rotation_matrix(rng):
    """
    Generate a uniformly distributed rotation in SO(3)
    using a normalized Gaussian quaternion.

    Quaternion convention:
        q = [w, x, y, z]
    """

    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)

    w, x, y, z = quat

    R = np.array([
        [
            1.0 - 2.0*(y*y + z*z),
            2.0*(x*y - z*w),
            2.0*(x*z + y*w)
        ],
        [
            2.0*(x*y + z*w),
            1.0 - 2.0*(x*x + z*z),
            2.0*(y*z - x*w)
        ],
        [
            2.0*(x*z - y*w),
            2.0*(y*z + x*w),
            1.0 - 2.0*(x*x + y*y)
        ]
    ])

    return R


def random_htm(rng, p_min, p_max):
    """
    Generate a random homogeneous transformation.

    Position:
        uniform inside the box [p_min, p_max]

    Orientation:
        uniform in SO(3)
    """

    H = np.eye(4)

    H[0:3, 0:3] = random_rotation_matrix(rng)
    H[0:3, 3] = rng.uniform(p_min, p_max)

    return H


# ============================================================
# Task function, Jr and Jrdot
# ============================================================

def task_function_second_order(robot, htm_tg, q, qdot):
    """
    Compute:

        r(q)
        Jr(q)
        Jrdot(q, qdot)

    for the UAIBot end-effector pose task function.

    Returns
    -------
    r : ndarray, shape (6, 1)
        Task function.

    Jr : ndarray, shape (6, n)
        Jacobian of the task function.

    Jrdot : ndarray, shape (6, n)
        Time derivative of Jr.
    """

    q = np.asarray(q, dtype=float).reshape(-1)
    qdot = np.asarray(qdot, dtype=float).reshape(-1)

    htm_tg = np.asarray(htm_tg, dtype=float)

    n = len(q)

    # --------------------------------------------------------
    # Target pose
    # --------------------------------------------------------

    p_des = htm_tg[0:3, 3]

    x_des = htm_tg[0:3, 0]
    y_des = htm_tg[0:3, 1]
    z_des = htm_tg[0:3, 2]

    # --------------------------------------------------------
    # Current geometric Jacobian and FK
    # --------------------------------------------------------

    Jg, htm_eef = robot.jac_geo(
        q=q,
        axis="eef"
    )

    Jg = np.asarray(Jg, dtype=float)
    htm_eef = np.asarray(htm_eef, dtype=float)

    # Analytical Jgdot
    Jgdot, _, _ = robot.compute_jgdot(
        q=q,
        qdot=qdot
    )

    Jgdot = np.asarray(Jgdot, dtype=float)

    # Split geometric Jacobian
    Jv = Jg[0:3, :]
    Jw = Jg[3:6, :]

    Jvdot = Jgdot[0:3, :]
    Jwdot = Jgdot[3:6, :]

    # --------------------------------------------------------
    # Current end-effector pose
    # --------------------------------------------------------

    p = htm_eef[0:3, 3]

    x = htm_eef[0:3, 0]
    y = htm_eef[0:3, 1]
    z = htm_eef[0:3, 2]

    # --------------------------------------------------------
    # Task function
    # --------------------------------------------------------

    r = np.zeros((6, 1))

    r[0:3, 0] = p - p_des

    # Keep the same definition used by UAIBot
    r[3, 0] = max(1.0 - x_des @ x, 0.0)
    r[4, 0] = max(1.0 - y_des @ y, 0.0)
    r[5, 0] = max(1.0 - z_des @ z, 0.0)

    # --------------------------------------------------------
    # Jr
    # --------------------------------------------------------

    Jr = np.zeros((6, n))

    # Position
    Jr[0:3, :] = Jv

    # Orientation
    Jr[3, :] = x_des @ skew(x) @ Jw
    Jr[4, :] = y_des @ skew(y) @ Jw
    Jr[5, :] = z_des @ skew(z) @ Jw

    # --------------------------------------------------------
    # Jrdot
    # --------------------------------------------------------

    Jrdot = np.zeros((6, n))

    # Position:
    #
    # Jr_position = Jv
    #
    # therefore
    #
    # Jrdot_position = Jvdot
    #
    Jrdot[0:3, :] = Jvdot

    # Angular velocity
    omega = Jw @ qdot

    # Time derivative of end-effector axes:
    #
    # adot = omega x a
    #
    #      = -S(a) omega
    #
    xdot = -skew(x) @ omega
    ydot = -skew(y) @ omega
    zdot = -skew(z) @ omega

    # For each orientation component:
    #
    # Jr_i = a_des^T S(a) Jw
    #
    # Therefore:
    #
    # Jrdot_i =
    #
    # a_des^T [
    #     S(adot) Jw
    #     +
    #     S(a) Jwdot
    # ]
    #

    Jrdot[3, :] = x_des @ (
        skew(xdot) @ Jw
        +
        skew(x) @ Jwdot
    )

    Jrdot[4, :] = y_des @ (
        skew(ydot) @ Jw
        +
        skew(y) @ Jwdot
    )

    Jrdot[5, :] = z_des @ (
        skew(zdot) @ Jw
        +
        skew(z) @ Jwdot
    )

    return r, Jr, Jrdot


# ============================================================
# Numerical Jr
# ============================================================

def compute_numerically_jr(robot, htm_tg, q, eps):
    """
    Compute Jr numerically using central differences:

                  r(q + eps e_i) - r(q - eps e_i)
        Jr[:,i] = ---------------------------------
                              2 eps
    """

    q = np.asarray(q, dtype=float).reshape(-1)

    n = len(q)

    Jr_num = np.zeros((6, n))

    qdot_zero = np.zeros(n)

    for i in range(n):

        dq = np.zeros(n)
        dq[i] = eps

        r_prev, _, _ = task_function_second_order(
            robot=robot,
            htm_tg=htm_tg,
            q=q - dq,
            qdot=qdot_zero
        )

        r_next, _, _ = task_function_second_order(
            robot=robot,
            htm_tg=htm_tg,
            q=q + dq,
            qdot=qdot_zero
        )

        Jr_num[:, i] = (
            (r_next - r_prev) / (2.0 * eps)
        ).reshape(6)

    return Jr_num


# ============================================================
# Numerical Jrdot
# ============================================================

def compute_numerically_jrdot(robot, htm_tg, q, qdot, dt):
    """
    Compute Jrdot numerically using a central difference
    along the direction qdot:

        Jrdot =
            Jr(q + qdot dt) - Jr(q - qdot dt)
            ---------------------------------
                         2 dt
    """

    _, Jr_prev, _ = task_function_second_order(
        robot=robot,
        htm_tg=htm_tg,
        q=q - qdot * dt,
        qdot=qdot
    )

    _, Jr_next, _ = task_function_second_order(
        robot=robot,
        htm_tg=htm_tg,
        q=q + qdot * dt,
        qdot=qdot
    )

    return (Jr_next - Jr_prev) / (2.0 * dt)


# ============================================================
# Error metric
# ============================================================

def relative_frobenius_error(reference, estimate):
    """
    Relative Frobenius error:

        ||reference - estimate||_F
        --------------------------
             ||reference||_F
    """

    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    norm_reference = np.linalg.norm(
        reference,
        ord="fro"
    )

    if norm_reference < 1e-12:
        return np.linalg.norm(
            reference - estimate,
            ord="fro"
        )

    return (
        np.linalg.norm(
            reference - estimate,
            ord="fro"
        )
        /
        norm_reference
    )


# ============================================================
# Monte Carlo validation
# ============================================================

robot = ub.Robot.create_franka_emika_3()

rng = np.random.default_rng()


# ------------------------------------------------------------
# Test parameters
# ------------------------------------------------------------

num_tests = 10000

# Numerical derivative used for Jrdot
dt = 1e-3

# Maximum acceptable relative error
error_threshold_percent = 0.1


# ------------------------------------------------------------
# Random target position box
# ------------------------------------------------------------

p_min = np.array([
    -0.8,
    -0.8,
    0.0
])

p_max = np.array([
    0.8,
    0.8,
    1.2
])


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

jrdot_success = 0

jrdot_errors_percent = []


# ============================================================
# Run tests
# ============================================================

for k in range(num_tests):

    # Random robot configuration
    q = rng.uniform(
        -np.pi,
        np.pi,
        size=7
    )

    # Random joint velocity
    qdot = rng.uniform(
        -np.pi,
        np.pi,
        size=7
    )

    # Random target in SE(3)
    htm_tg = random_htm(
        rng=rng,
        p_min=p_min,
        p_max=p_max
    )

    # --------------------------------------------------------
    # C++ analytical Jrdot
    # --------------------------------------------------------

    _, _, Jrdot_cpp = robot.compute_jrdot(
        htm_tg=htm_tg,
        q=q,
        qdot=qdot
    )

    # --------------------------------------------------------
    # Numerical Jrdot
    # --------------------------------------------------------

    Jrdot_num = compute_numerically_jrdot(
        robot=robot,
        htm_tg=htm_tg,
        q=q,
        qdot=qdot,
        dt=dt
    )

    jrdot_error = relative_frobenius_error(
        Jrdot_num,
        Jrdot_cpp
    )

    jrdot_error_percent = 100.0 * jrdot_error

    jrdot_errors_percent.append(
        jrdot_error_percent
    )

    if jrdot_error_percent < error_threshold_percent:
        jrdot_success += 1


# ============================================================
# Results
# ============================================================

jrdot_success_rate = (
    100.0 * jrdot_success / num_tests
)


print()
print("=" * 60)
print("Jrdot validation")
print("=" * 60)

print(f"Number of tests: {num_tests}")
print(f"Threshold: {error_threshold_percent:.6f}%")

print(
    f"Successes: "
    f"{jrdot_success}/{num_tests}"
)

print(
    f"Success rate: "
    f"{jrdot_success_rate:.2f}%"
)

print()

print(
    f"Mean relative error: "
    f"{np.mean(jrdot_errors_percent):.6e}%"
)

print(
    f"Maximum relative error: "
    f"{np.max(jrdot_errors_percent):.6e}%"
)

print(
    f"Minimum relative error: "
    f"{np.min(jrdot_errors_percent):.6e}%"
)
