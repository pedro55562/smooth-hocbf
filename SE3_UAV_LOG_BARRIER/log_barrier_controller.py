from dataclasses import dataclass
from time import perf_counter

import casadi as ca
import numpy as np
from scipy.optimize import linprog

from constraint_builder import ConstraintSet


@dataclass(frozen=True)
class LogBarrierConfig:
    mu_cbf: float = 1e-3
    mu_twist_limits: float = 1e-3
    interior_margin: float = 1e-6
    ipopt_max_iter: int = 80
    ipopt_tol: float = 1e-8
    ipopt_acceptable_tol: float = 1e-6


@dataclass
class LogBarrierResult:
    twist: np.ndarray
    distances: list[float]
    slacks: np.ndarray
    min_slack: float
    critical_constraint_name: str
    solve_time: float
    solver_iterations: int
    solver_success: bool
    solver_status: str
    nominal_cost: float
    barrier_cost: float
    total_cost: float


class LogBarrierController:
    """Direct spatial-twist safety filter using log barriers in the objective."""

    def __init__(
        self,
        constraint_count: int,
        cbf_count: int,
        config: LogBarrierConfig,
    ):
        self.constraint_count = int(constraint_count)
        self.cbf_count = int(cbf_count)
        self.config = config
        self.solver = self._create_solver()

    @staticmethod
    def _as_column(value, size: int) -> np.ndarray:
        return np.asarray(value, dtype=float).reshape(size, 1)

    def _create_solver(self):
        xi = ca.MX.sym("xi", 6)
        param_size = 6 + self.constraint_count * 6 + self.constraint_count
        params = ca.MX.sym("params", param_size)

        xi_d = params[0:6]
        offset = 6
        A_flat = params[offset : offset + self.constraint_count * 6]
        offset += self.constraint_count * 6
        b = params[offset : offset + self.constraint_count]

        diff = xi - xi_d
        objective = 0.5 * ca.dot(diff, diff)

        for row_index in range(self.constraint_count):
            row_offset = row_index * 6
            row = A_flat[row_offset : row_offset + 6]
            slack = ca.dot(row, xi) - b[row_index]
            mu = (
                self.config.mu_cbf
                if row_index < self.cbf_count
                else self.config.mu_twist_limits
            )
            objective += -mu * ca.log(slack)

        nlp = {"x": xi, "f": objective, "p": params}
        options = {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": self.config.ipopt_max_iter,
            "ipopt.tol": self.config.ipopt_tol,
            "ipopt.acceptable_tol": self.config.ipopt_acceptable_tol,
            "ipopt.warm_start_init_point": "yes",
        }
        return ca.nlpsol("log_barrier_solver", "ipopt", nlp, options)

    def _pack_parameters(
        self,
        nominal_twist: np.ndarray,
        constraints: ConstraintSet,
    ) -> np.ndarray:
        A = np.asarray(constraints.A, dtype=float).reshape(self.constraint_count, 6)
        b = np.asarray(constraints.b, dtype=float).reshape(self.constraint_count, 1)
        nominal_twist = self._as_column(nominal_twist, 6)
        return np.vstack((nominal_twist, A.reshape(-1, 1), b)).reshape(-1)

    @staticmethod
    def _slack(A: np.ndarray, b: np.ndarray, xi: np.ndarray) -> np.ndarray:
        return A @ xi - b

    def _is_strictly_feasible(
        self,
        A: np.ndarray,
        b: np.ndarray,
        xi: np.ndarray,
    ) -> bool:
        slack = self._slack(A, b, xi)
        return bool(np.all(np.isfinite(slack)) and np.all(slack > self.config.interior_margin))

    def _scaled_candidates(self, vector: np.ndarray):
        alphas = [1.0, 0.99, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0]
        for alpha in alphas:
            yield alpha * vector

    def _find_interior_with_lp(
        self,
        A: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray | None:
        c = np.zeros(7)
        c[-1] = -1.0

        A_ub = np.hstack((-A, np.ones((A.shape[0], 1))))
        b_ub = -b.reshape(-1)
        bounds = [(None, None)] * 7

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not result.success:
            return None

        xi = result.x[:6].reshape(6, 1)
        max_min_slack = float(result.x[-1])
        if max_min_slack <= self.config.interior_margin:
            return None
        return xi

    def _interior_guess(
        self,
        A: np.ndarray,
        b: np.ndarray,
        preferred_guess: np.ndarray,
        nominal_twist: np.ndarray,
    ) -> np.ndarray:
        candidates = []
        candidates.extend(self._scaled_candidates(preferred_guess))
        candidates.extend(self._scaled_candidates(nominal_twist))
        candidates.append(np.zeros((6, 1)))

        for candidate in candidates:
            candidate = self._as_column(candidate, 6)
            if self._is_strictly_feasible(A, b, candidate):
                return candidate

        lp_guess = self._find_interior_with_lp(A, b)
        if lp_guess is not None and self._is_strictly_feasible(A, b, lp_guess):
            return lp_guess

        zero_slack = self._slack(A, b, np.zeros((6, 1)))
        raise RuntimeError(
            "Could not find a strictly feasible log-barrier initial guess. "
            f"Minimum slack at xi=0 is {float(np.min(zero_slack)):.6e}."
        )

    def _costs(
        self,
        xi: np.ndarray,
        nominal_twist: np.ndarray,
        slacks: np.ndarray,
    ) -> tuple[float, float, float]:
        diff = xi - nominal_twist
        nominal_cost = 0.5 * float((diff.T @ diff).item())

        slack_vector = np.asarray(slacks, dtype=float).reshape(-1)
        cbf_slacks = slack_vector[: self.cbf_count]
        limit_slacks = slack_vector[self.cbf_count :]
        barrier_cost = 0.0
        if cbf_slacks.size:
            barrier_cost -= self.config.mu_cbf * float(np.sum(np.log(cbf_slacks)))
        if limit_slacks.size:
            barrier_cost -= self.config.mu_twist_limits * float(
                np.sum(np.log(limit_slacks))
            )

        total_cost = nominal_cost + barrier_cost
        return nominal_cost, barrier_cost, total_cost

    def solve(
        self,
        nominal_twist: np.ndarray,
        constraints: ConstraintSet,
        initial_guess: np.ndarray,
    ) -> LogBarrierResult:
        nominal_twist = self._as_column(nominal_twist, 6)
        A = np.asarray(constraints.A, dtype=float).reshape(self.constraint_count, 6)
        b = np.asarray(constraints.b, dtype=float).reshape(self.constraint_count, 1)
        guess = self._interior_guess(A, b, initial_guess, nominal_twist)
        params = self._pack_parameters(nominal_twist, constraints)

        start = perf_counter()
        solution = self.solver(x0=guess, p=params)
        solve_time = perf_counter() - start

        stats = self.solver.stats()
        solver_success = bool(stats.get("success", False))
        solver_status = str(stats.get("return_status", "unknown"))
        solver_iterations = int(stats.get("iter_count", -1))

        xi = self._as_column(np.asarray(solution["x"], dtype=float), 6)
        slacks = self._slack(A, b, xi)
        slack_vector = np.asarray(slacks, dtype=float).reshape(-1)

        if not solver_success:
            raise RuntimeError(f"IPOPT failed with status '{solver_status}'.")
        if not np.all(np.isfinite(xi)) or not np.all(np.isfinite(slack_vector)):
            raise RuntimeError("IPOPT returned non-finite values.")
        if np.any(slack_vector <= 0.0):
            raise RuntimeError(
                f"Log-barrier solution left the domain. Minimum slack is {float(np.min(slack_vector)):.6e}."
            )

        critical_index = int(np.argmin(slack_vector))
        nominal_cost, barrier_cost, total_cost = self._costs(
            xi, nominal_twist, slack_vector
        )

        return LogBarrierResult(
            twist=xi,
            distances=constraints.distances,
            slacks=slacks,
            min_slack=float(slack_vector[critical_index]),
            critical_constraint_name=constraints.names[critical_index],
            solve_time=solve_time,
            solver_iterations=solver_iterations,
            solver_success=solver_success,
            solver_status=solver_status,
            nominal_cost=nominal_cost,
            barrier_cost=barrier_cost,
            total_cost=total_cost,
        )
