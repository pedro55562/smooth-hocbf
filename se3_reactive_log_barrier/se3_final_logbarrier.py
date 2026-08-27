import os

import numpy as np
import uaibot as ub
from scipy.linalg import expm
from scipy.optimize import linprog

from aux_functions import log_SE3
from setup import create_scenario, plot_dist_min, plot_pose_error, plot_u_xi

try:
    import casadi as ca
except ImportError as exc:
    raise ImportError(
        "IPOPT requires CasADi. Install it with: pip install casadi"
    ) from exc


LOG_BARRIER_SCALE = 1e4
POSE_TOLERANCE = 0.025


class LogBarrierControlProblem:
    def __init__(self, num_constraints, mu):
        self.mu = mu
        self.num_constraints = num_constraints
        self.previous_u = None

        if not ca.has_nlpsol("ipopt"):
            raise RuntimeError(
                "The IPOPT plugin is not available in this CasADi installation."
            )

        u = ca.MX.sym("u", 6)
        parameter_size = 6 + num_constraints * 6 + num_constraints
        parameters = ca.MX.sym("parameters", parameter_size)
        u_d = parameters[:6]

        objective = ca.sumsqr(u - u_d)
        A_offset = 6
        b_offset = A_offset + num_constraints * 6
        for row_index in range(num_constraints):
            row_offset = A_offset + row_index * 6
            row = parameters[row_offset : row_offset + 6]
            slack = ca.dot(row, u) - parameters[b_offset + row_index]
            objective -= mu * ca.log(LOG_BARRIER_SCALE * slack)

        options = {
            "print_time": False,
            "error_on_fail": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.tol": 1e-8,
            "ipopt.max_iter": 1000,
            "ipopt.warm_start_init_point": "yes",
        }
        self.solver = ca.nlpsol(
            "log_barrier_solver",
            "ipopt",
            {"x": u, "f": objective, "p": parameters},
            options,
        )

        self.solve_count = 0
        self.success_count = 0
        self.status_list = []
        self.min_slack_list = []

    @staticmethod
    def _is_strictly_feasible(A, b, u):
        slack = A @ u - b
        return bool(np.all(np.isfinite(slack)) and np.all(slack > 0))

    def _initial_guess(self, A, b, u_d):
        if self.previous_u is not None and self._is_strictly_feasible(
            A, b, self.previous_u
        ):
            return self.previous_u

        if self._is_strictly_feasible(A, b, u_d):
            return u_d

        objective = np.zeros(7)
        objective[-1] = -1.0
        A_ub = np.hstack((-A, np.ones((self.num_constraints, 1))))
        result = linprog(
            objective,
            A_ub=A_ub,
            b_ub=-b.reshape(-1),
            bounds=[(None, None)] * 7,
            method="highs",
        )
        if result.success:
            interior_u = result.x[:6].reshape((6, 1))
            if self._is_strictly_feasible(A, b, interior_u):
                return interior_u

        return u_d

    def solve(self, A, b, u_d):
        A_value = np.asarray(A, dtype=float)
        b_value = np.asarray(b, dtype=float).reshape((-1, 1))
        u_d_value = np.asarray(u_d, dtype=float).reshape((6, 1))

        self.solve_count += 1
        parameters = np.concatenate(
            (u_d_value.reshape(-1), A_value.reshape(-1), b_value.reshape(-1))
        )
        initial_guess = self._initial_guess(A_value, b_value, u_d_value)

        try:
            solution = self.solver(x0=initial_guess, p=parameters)
            solver_stats = self.solver.stats()
            status = str(solver_stats.get("return_status", "unknown"))
        except Exception as exc:
            status = f"solver_exception:{type(exc).__name__}"
            self.status_list.append(status)
            return None, f"{status}: {exc}", None

        self.status_list.append(status)
        if status == "Solved_To_Acceptable_Level":
            print(f"IPOPT solve {self.solve_count} returned status: {status}")

        u_array = np.asarray(solution["x"], dtype=float).reshape((6, 1))
        u_value = np.matrix(u_array)
        slack = A_value @ np.asarray(u_value) - b_value
        min_slack = float(np.min(slack))
        self.min_slack_list.append(min_slack)

        if not solver_stats.get("success", False):
            return None, status, min_slack

        if not np.all(np.isfinite(u_array)) or min_slack <= 0:
            failure_status = f"{status}_domain_violation"
            self.status_list[-1] = failure_status
            return None, failure_status, min_slack

        self.previous_u = u_array
        self.success_count += 1
        return u_value, status, min_slack


def cmpt_lambda_terms(objA, objB, H, xi, h, eps):
    sA = H[0:3, -1]
    vA = xi[0:3, -1]
    wA = xi[3:6, -1]

    htm_A = np.matrix(objA.htm)
    objA.set_ani_frame(H)

    a_star, b_star, lambda_AB, _ = objA.compute_dist(
        obj=objB,
        h=h,
        eps=eps,
        tol=1e-6,
        no_iter_max=6000,
    )

    D_xi_lambda_AB = np.matrix(np.zeros((1, 6)))
    D_xi_lambda_AB[:, 0:3] = (a_star - b_star).T / (1e-6 + lambda_AB)
    D_xi_lambda_AB[:, 3:6] = (
        ub.Utils.S(b_star - sA) * (a_star - b_star)
    ).T / (1e-6 + lambda_AB)

    psi_A_a = vA + ub.Utils.S(wA) * (a_star - sA)
    psi_A_b = vA + ub.Utils.S(wA) * (b_star - sA)
    J_Pi_A = np.matrix(objA.cpp_obj.projection_jacobian(b_star, h, eps))
    J_Pi_B = np.matrix(objB.cpp_obj.projection_jacobian(a_star, h, eps))
    d_Pi_A_dt = psi_A_a - J_Pi_A * psi_A_b

    a_star_dot = np.linalg.inv(np.identity(3) - J_Pi_A * J_Pi_B) * d_Pi_A_dt
    b_star_dot = J_Pi_B * a_star_dot

    d_D_xi_Lambda_AB_dt = np.matrix(np.zeros((1, 6)))
    d_D_xi_Lambda_AB_dt[:, 0:3] = (a_star_dot - b_star_dot).T
    d_D_xi_Lambda_AB_dt[:, 3:6] = (
        ub.Utils.S(b_star - sA) * (a_star_dot - b_star_dot)
        + ub.Utils.S(b_star_dot - vA) * (a_star - b_star)
    ).T

    d_D_xi_lambda_AB_dt = (
        d_D_xi_Lambda_AB_dt
        - (D_xi_lambda_AB * xi) * D_xi_lambda_AB
    ) / (1e-6 + lambda_AB)

    objA.set_ani_frame(htm_A)
    return lambda_AB, D_xi_lambda_AB, d_D_xi_lambda_AB_dt


def cmpt_control(
    H,
    xi,
    obj_robot,
    list_obs,
    u_d,
    h=0.05,
    eps=0.01,
    eta=0.3,
    lambda_min=0.01,
    xi_lim=0.08,
    optimizer=None,
):
    falhou = False
    A = np.matrix(np.zeros((0, 6)))
    b = np.matrix(np.zeros((0, 1)))

    for obs in list_obs:
        lambda_RO, D_xi_lambda_RO, d_D_xi_lambda_RO_dt = cmpt_lambda_terms(
            obj_robot, obs, H, xi, h, eps
        )

        ff = -d_D_xi_lambda_RO_dt * xi
        d_lambda_RO_dt = D_xi_lambda_RO * xi
        b_temp = (
            ff
            - 2 * eta * d_lambda_RO_dt
            - eta * eta * (lambda_RO - lambda_min)
        )

        A = np.vstack([A, D_xi_lambda_RO])
        b = np.vstack([b, b_temp])

    A = np.vstack([A, np.identity(6)])
    b = np.vstack([b, -xi_lim * np.ones((6, 1))])
    A = np.vstack([A, -np.identity(6)])
    b = np.vstack([b, -xi_lim * np.ones((6, 1))])

    try:
        if optimizer is None:
            raise ValueError("optimizer must be a LogBarrierControlProblem")
        u, status, min_slack = optimizer.solve(A, b, u_d)
        if u is None:
            raise RuntimeError(
                f"IPOPT status: {status}, min_slack: {min_slack}"
            )
    except Exception as exc:
        falhou = True
        print("\n Log barrier falhou!  ")
        print("Motivo: ", exc)
        sim.save(
            address=os.path.dirname(__file__),
            file_name="se3_reactive_log_barrier",
        )
        return 0, falhou

    return u, falhou


def cmpt_control_reactive(H, xi, H_d, kc=0.5):
    def ext(M):
        return np.matrix(np.diag(M)).T

    s = H[0:3, -1]
    s_d = H_d[0:3, -1]
    Q = H[0:3, 0:3]
    Q_d = H_d[0:3, 0:3]

    w = xi[3:6, -1]
    Sx = ub.Utils.S([1, 0, 0])
    Sy = ub.Utils.S([0, 1, 0])
    Sz = ub.Utils.S([0, 0, 1])
    Sw = ub.Utils.S(w)

    r = np.matrix(np.zeros((6, 1)))
    r[0:3, -1] = s - s_d
    r[3:6, -1] = ext(np.identity(3) - Q_d.T * Q)

    D_xi_r = np.matrix(np.zeros((6, 6)))
    D_xi_r[0:3, :] = np.hstack([np.identity(3), np.zeros((3, 3))])
    ax = ext(-Q_d.T * Sx * Q)
    ay = ext(-Q_d.T * Sy * Q)
    az = ext(-Q_d.T * Sz * Q)
    D_xi_r[3:6, :] = np.hstack([np.zeros((3, 3)), ax, ay, az])

    d_D_xi_r_dt = np.matrix(np.zeros((6, 6)))
    dax = ext(-Q_d.T * Sx * Sw * Q)
    day = ext(-Q_d.T * Sy * Sw * Q)
    daz = ext(-Q_d.T * Sz * Sw * Q)
    d_D_xi_r_dt[3:6, :] = np.hstack(
        [np.zeros((3, 3)), dax, day, daz]
    )

    u_d = ub.Utils.dp_inv(D_xi_r) * (
        -d_D_xi_r_dt * xi
        - 2 * kc * D_xi_r * xi
        - kc * kc * r
    )
    return u_d, r


def propagate_htm(htm, xi, dt_step):
    p = htm[0:3, 3].reshape(3, 1)
    R = htm[0:3, 0:3]

    v = np.asarray(xi[0:3]).reshape(3, 1)
    w = np.asarray(xi[3:6]).reshape(3, 1)

    R_next = expm(ub.Utils.S(w) * dt_step) @ R
    p_next = p + v * dt_step

    htm_next = np.eye(4)
    htm_next[0:3, 0:3] = R_next
    htm_next[0:3, 3] = p_next.flatten()
    return np.matrix(htm_next)


##############################
#     Scenario Setup         #
##############################

scenario = create_scenario()
sim = scenario.sim
robot_body = scenario.robot_body
robot_body_copy = scenario.robot_body_copy
robot_UAV = scenario.robot_UAV
all_obs = scenario.all_obs
H_d = scenario.H_d
H = scenario.H
xi = scenario.xi
dt = scenario.dt
t_max = scenario.t_max
simular_movimento = scenario.simular_movimento

##############################
#     Control Parameters     #
##############################

param_eta = 1.2
param_obs_delta = 0.01
mu = 5e-4

use_generalized_distance = True

if use_generalized_distance:
    dist_param_h = 0.05
    dist_param_eps = 0.03
else:
    dist_param_h = 0
    dist_param_eps = 0

num_control_constraints = len(all_obs) + 12
control_optimizer = LogBarrierControlProblem(num_control_constraints, mu)

##############################
#     Simulation Settings    #
##############################

min_ec_dist = []
xi_list = []
u_list = []
time_list = []
error = []
last_err = np.linalg.norm(log_SE3(ub.Utils.inv_htm(H) @ H_d))

##############################
#      Simulation Loop       #
##############################

falhou = False

if simular_movimento:
    for k in range(int(t_max / dt)):
        if last_err < POSE_TOLERANCE:
            print("last d(H) : ", last_err)
            break

        t = k * dt
        ud, _ = cmpt_control_reactive(H, xi, H_d, kc=0.5)

        u, falhou = cmpt_control(
            H,
            xi,
            robot_body_copy,
            all_obs,
            ud,
            dist_param_h,
            dist_param_eps,
            param_eta,
            param_obs_delta,
            xi_lim=1,
            optimizer=control_optimizer,
        )
        if falhou:
            break

        xi = xi + u * dt
        H = propagate_htm(H, xi, dt)
        robot_UAV.add_ani_frame(time=t, htm=H)

        ec_dist = []
        for obs in all_obs:
            _, _, dist, _ = robot_body.compute_dist(obs)
            ec_dist.append(dist)

        min_ec_dist.append(min(ec_dist))
        u_list.append(u)
        xi_list.append(xi)
        time_list.append(t)
        error.append(np.linalg.norm(log_SE3(ub.Utils.inv_htm(H) @ H_d)))
        last_err = error[-1]

##############################
#          Results           #
##############################

print("Log barrier mu: ", mu)
print(
    "IPOPT successful solves: ",
    f"{control_optimizer.success_count}/{control_optimizer.solve_count}",
)
if error:
    print("Final d(H): ", last_err)
    print("Converged to target tolerance: ", last_err < POSE_TOLERANCE)
if control_optimizer.min_slack_list:
    print(
        "Minimum log-barrier domain slack: ",
        min(control_optimizer.min_slack_list),
    )
status_counts = {
    status: control_optimizer.status_list.count(status)
    for status in set(control_optimizer.status_list)
}
print("IPOPT status counts: ", status_counts)

np.savez(
    os.path.join(os.path.dirname(__file__), "control_data.npz"),
    u=np.asarray(u_list).reshape(-1, 6),
    xi=np.asarray(xi_list).reshape(-1, 6),
    time=np.asarray(time_list),
    pose_error=np.asarray(error),
    min_distance=np.asarray(min_ec_dist),
    solver_status=np.asarray(control_optimizer.status_list),
    solver_min_slack=np.asarray(control_optimizer.min_slack_list),
    dt=dt,
)

sim.save(
    address=os.path.dirname(__file__),
    file_name="se3_reactive_log_barrier",
)

plot_dist_min(
    min_ec_dist,
    time_list,
    file_name="minimum_distance.pdf",
    labels=None,
    xlabel="Time (s)",
    ylabel=r"$d_{min}$",
    show_plot=False,
    title=None,
)

plot_u_xi(
    u_list,
    xi_list,
    time_list,
    file_name="u_xi_combined.pdf",
    labels_u=[
        r"$u_{v_x}$",
        r"$u_{v_y}$",
        r"$u_{v_z}$",
        r"$u_{\omega_x}$",
        r"$u_{\omega_y}$",
        r"$u_{\omega_z}$",
    ],
    labels_xi=[
        r"$v_x$",
        r"$v_y$",
        r"$v_z$",
        r"$\omega_x$",
        r"$\omega_y$",
        r"$\omega_z$",
    ],
)

plot_pose_error(
    error,
    time_list,
    file_name="pose_error.pdf",
    tolerance=POSE_TOLERANCE,
    show_plot=False,
)
