from dataclasses import dataclass
from typing import Iterable

import numpy as np
import uaibot as ub


LinearConstraint = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class CBFConfig:
    eta: float = 1.2
    min_distance: float = 0.01
    distance_h: float = 0.05
    distance_eps: float = 0.03
    twist_limit: float = 1.0
    distance_regularization: float = 1e-6
    distance_tol: float = 1e-6
    distance_max_iter: int = 6000


@dataclass
class QPResult:
    twist: np.ndarray
    distances: list[float]
    A: np.ndarray
    b: np.ndarray


class VelocityCBFController:
    """QP safety filter for a spatial twist.

    The solver convention used by ``ub.Utils.solve_qp`` is kept from the
    original code: linear inequalities are assembled as

        A @ xi >= b.

    The CBF is the first-order counterpart of the original explicit CBF:

        h = lambda - lambda_min
        h_dot + eta h >= 0

    which gives

        D_xi(lambda) @ xi >= -eta (lambda - lambda_min).
    """

    def __init__(self, robot_geometry, obstacles: Iterable, config: CBFConfig):
        self.robot_geometry = robot_geometry
        self.obstacles = list(obstacles)
        self.config = config

    @staticmethod
    def _as_column(value, size: int) -> np.ndarray:
        return np.asarray(value, dtype=float).reshape(size, 1)

    def _distance_and_gradient(self, H: np.ndarray, obstacle) -> tuple[float, np.ndarray]:
        cfg = self.config
        H = np.asarray(H, dtype=float).reshape(4, 4)
        robot_htm_before = np.asarray(self.robot_geometry.htm, dtype=float).copy()

        try:
            self.robot_geometry.set_ani_frame(H)
            point_robot, point_obs, distance, _ = self.robot_geometry.compute_dist(
                obj=obstacle,
                h=cfg.distance_h,
                eps=cfg.distance_eps,
                tol=cfg.distance_tol,
                no_iter_max=cfg.distance_max_iter,
            )
        finally:
            self.robot_geometry.set_ani_frame(robot_htm_before)

        point_robot = self._as_column(point_robot, 3)
        point_obs = self._as_column(point_obs, 3)
        position = H[:3, 3].reshape(3, 1)

        direction = point_robot - point_obs
        denominator = float(distance) + cfg.distance_regularization

        grad_linear = direction.T / denominator
        grad_angular = (
            np.asarray(ub.Utils.S(point_obs - position), dtype=float) @ direction
        ).T / denominator

        gradient = np.hstack((grad_linear, grad_angular))
        return float(distance), gradient

    def _cbf_constraints(self, H: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
        rows = []
        bounds = []
        distances = []

        for obstacle in self.obstacles:
            distance, gradient = self._distance_and_gradient(H, obstacle)

            # First-order CBF:
            # D_lambda @ xi + eta * (lambda - lambda_min) >= 0
            rhs = -self.config.eta * (distance - self.config.min_distance)

            rows.append(gradient)
            bounds.append([[rhs]])
            distances.append(distance)

        if not rows:
            return np.empty((0, 6)), np.empty((0, 1)), distances

        return np.vstack(rows), np.vstack(bounds), distances

    def _twist_constraints(self) -> LinearConstraint:
        limit = float(self.config.twist_limit)
        identity = np.eye(6)

        # -limit <= xi_i <= limit written as A @ xi >= b
        A = np.vstack((identity, -identity))
        b = -limit * np.ones((12, 1))
        return A, b

    @staticmethod
    def _stack_constraints(constraints: Iterable[LinearConstraint]) -> LinearConstraint:
        A_blocks = []
        b_blocks = []

        for A, b in constraints:
            A = np.asarray(A, dtype=float).reshape(-1, 6)
            b = np.asarray(b, dtype=float).reshape(-1, 1)
            if A.shape[0] != b.shape[0]:
                raise ValueError("A and b must have the same number of rows")
            if A.shape[0] > 0:
                A_blocks.append(A)
                b_blocks.append(b)

        if not A_blocks:
            return np.empty((0, 6)), np.empty((0, 1))

        return np.vstack(A_blocks), np.vstack(b_blocks)

    def solve(
        self,
        H: np.ndarray,
        nominal_twist: np.ndarray,
        extra_constraints: Iterable[LinearConstraint] | None = None,
    ) -> QPResult:
        """Filter a nominal twist with explicit linear constraints.

        ``extra_constraints`` is intentionally part of the interface so new
        conventional constraints can be added later without changing the CBF
        implementation. Each item must be a pair ``(A_i, b_i)`` representing
        ``A_i @ xi >= b_i``.
        """
        nominal_twist = self._as_column(nominal_twist, 6)

        A_cbf, b_cbf, distances = self._cbf_constraints(H)
        A_lim, b_lim = self._twist_constraints()

        constraints = [(A_cbf, b_cbf), (A_lim, b_lim)]
        if extra_constraints is not None:
            constraints.extend(extra_constraints)

        A, b = self._stack_constraints(constraints)

        H_qp = 2.0 * np.eye(6)
        f_qp = -2.0 * nominal_twist

        # UAIBot still uses np.matrix semantics internally in parts of Utils.
        # Keep the conversion isolated at this boundary; the rest of the code
        # uses ndarrays and explicit @ multiplication.
        twist = ub.Utils.solve_qp(
            np.matrix(H_qp),
            np.matrix(f_qp),
            np.matrix(A),
            np.matrix(b),
        )
        twist = self._as_column(twist, 6)

        return QPResult(twist=twist, distances=distances, A=A, b=b)
