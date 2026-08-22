import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Simulation parameters
# ============================================================

dt = 0.01
t = 0.0
tmax = 6.0

# Pole of the critically damped task dynamics
lam = 2.0

# Damping for the pseudoinverse
damping = 1e-3


# ============================================================
# Robot
# ============================================================

robot = ub.Robot.create_franka_emika_3()

n = len(robot.q)

# Initial state
q = np.asarray(robot.q, dtype=float).reshape(n)
qdot = np.zeros(n)


# ============================================================
# Desired end-effector pose
# ============================================================

htm_d = robot.fkm() * ub.Utils.trn([0.5, 0.1, .2])*ub.Utils.roty(np.pi/2)

frame_d = ub.Frame(htm=htm_d)

sim = ub.Simulation([
    robot,
    frame_d
])


# ============================================================
# Histories
# ============================================================

hist_r = []
hist_qdot = []
hist_qddot = []
hist_t = []


# ============================================================
# Main control loop
# ============================================================

for i in range(round(tmax / dt)):

    # --------------------------------------------------------
    # Task function and its differential kinematics
    # --------------------------------------------------------

    r, Jr, Jrdot = robot.compute_jrdot(
        htm_tg=htm_d,
        q=q,
        qdot=qdot
    )

    r = np.asarray(r, dtype=float).reshape(6, 1)
    Jr = np.asarray(Jr, dtype=float)
    Jrdot = np.asarray(Jrdot, dtype=float)

    qdot_col = qdot.reshape(n, 1)


    # --------------------------------------------------------
    # First derivative of the task function
    #
    # rdot = Jr qdot
    # --------------------------------------------------------

    rdot = Jr @ qdot_col


    # --------------------------------------------------------
    # Desired critically damped second-order dynamics
    #
    # rddot + 2 lam rdot + lam² r = 0
    #
    # Therefore:
    #
    # rddot_des = -2 lam rdot - lam² r
    # --------------------------------------------------------

    rddot_des = (
        -2.0 * lam * rdot
        -lam**2 * r
    )


    # --------------------------------------------------------
    # Actual task acceleration:
    #
    # rddot = Jr qddot + Jrdot qdot
    #
    # Therefore:
    #
    # Jr qddot = rddot_des - Jrdot qdot
    # --------------------------------------------------------

    rhs = (
        rddot_des
        - Jrdot @ qdot_col
    )


    # --------------------------------------------------------
    # Joint acceleration
    # --------------------------------------------------------

    Jr_pinv = np.asarray(
        ub.Utils.dp_inv(Jr, damping),
        dtype=float
    )

    qddot = Jr_pinv @ rhs
    qddot = qddot.reshape(n)


    # --------------------------------------------------------
    # Store history
    # --------------------------------------------------------

    hist_r.append(r.reshape(6))
    hist_qdot.append(qdot.copy())
    hist_qddot.append(qddot.copy())
    hist_t.append(t)


    # --------------------------------------------------------
    # Integrate second-order joint kinematics
    #
    # Assuming constant qddot during dt:
    #
    # q(t+dt) =
    #     q + qdot dt + 1/2 qddot dt²
    #
    # qdot(t+dt) =
    #     qdot + qddot dt
    # --------------------------------------------------------

    q_next = (
        q
        + qdot * dt
        + 0.5 * qddot * dt**2
    )

    qdot_next = (
        qdot
        + qddot * dt
    )


    # --------------------------------------------------------
    # Send configuration to animation
    # --------------------------------------------------------

    robot.add_ani_frame(
        time=t + dt,
        q=q_next
    )


    # Update state
    q = q_next
    qdot = qdot_next

    t += dt


# ============================================================
# Convert histories to numpy arrays
# ============================================================

hist_r = np.asarray(hist_r)
hist_qdot = np.asarray(hist_qdot)
hist_qddot = np.asarray(hist_qddot)
hist_t = np.asarray(hist_t)

sim.save(
    address="/home/pedro/Projects/smooth-hocbf/manipulator/",
    file_name="teste",
)

# ============================================================
# Plot da função de tarefa
# ============================================================

plt.figure()
for i in range(hist_r.shape[1]):
    plt.plot(hist_t, hist_r[:, i], label=f"r{i+1}")

plt.xlabel("Time [s]")
plt.ylabel("Task function")
plt.title("Task function history")
plt.grid(True)
plt.legend()
plt.show()


# ============================================================
# Plot das velocidades articulares
# ============================================================

plt.figure()
for i in range(hist_qdot.shape[1]):
    plt.plot(hist_t, hist_qdot[:, i], label=f"qdot_{i+1}")

plt.xlabel("Time [s]")
plt.ylabel("Joint velocity [rad/s]")
plt.title("Joint velocity history")
plt.grid(True)
plt.legend()
plt.show()


# ============================================================
# Plot das acelerações articulares
# ============================================================

plt.figure()
for i in range(hist_qddot.shape[1]):
    plt.plot(hist_t, hist_qddot[:, i], label=f"qddot_{i+1}")

plt.xlabel("Time [s]")
plt.ylabel("Joint acceleration [rad/s²]")
plt.title("Joint acceleration history")
plt.grid(True)
plt.legend()
plt.show()