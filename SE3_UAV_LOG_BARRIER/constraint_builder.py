from dataclasses import dataclass
from typing import Iterable

import numpy as np
import uaibot as ub


@dataclass(frozen=True)
class ConstraintBuilderConfig:
    eta: float = 1.2
    min_distance: float = 0.01
    distance_h: float = 0.05
    distance_eps: float = 0.03
    twist_limit: float = 1.0
    distance_regularization: float = 1e-6
    distance_tol: float = 1e-6
    distance_max_iter: int = 6000


@dataclass
class ConstraintSet:
    A: np.ndarray
    b: np.ndarray
    names: list[str]
    distances: list[float]


class VelocityConstraintBuilder:
    """Build affine inequalities in the convention A @ xi >= b."""

    def __init__(self, robot_geometry, obstacles: Iterable, config: ConstraintBuilderConfig):
        self.robot_geometry = robot_geometry
        self.obstacles = list(obstacles)
        self.config = config

    @property
    def cbf_count(self) -> int:
        return len(self.obstacles)

    @property
    def constraint_count(self) -> int:
        return self.cbf_count + 12

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

    def _cbf_constraints(self, H: np.ndarray) -> ConstraintSet:
        rows = []
        bounds = []
        names = []
        distances = []

        for index, obstacle in enumerate(self.obstacles):
            distance, gradient = self._distance_and_gradient(H, obstacle)
            rhs = -self.config.eta * (distance - self.config.min_distance)

            rows.append(gradient)
            bounds.append([[rhs]])
            names.append(f"cbf_obstacle_{index}")
            distances.append(distance)

        if not rows:
            return ConstraintSet(
                A=np.empty((0, 6)),
                b=np.empty((0, 1)),
                names=[],
                distances=distances,
            )

        return ConstraintSet(
            A=np.vstack(rows),
            b=np.vstack(bounds),
            names=names,
            distances=distances,
        )

    def _twist_constraints(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        limit = float(self.config.twist_limit)
        identity = np.eye(6)
        labels = ["vx", "vy", "vz", "wx", "wy", "wz"]

        A = np.vstack((identity, -identity))
        b = -limit * np.ones((12, 1))
        names = (
            [f"twist_{label}_lower" for label in labels]
            + [f"twist_{label}_upper" for label in labels]
        )
        return A, b, names

    def build(self, H: np.ndarray) -> ConstraintSet:
        cbf = self._cbf_constraints(H)
        A_lim, b_lim, limit_names = self._twist_constraints()

        A = np.vstack((cbf.A, A_lim))
        b = np.vstack((cbf.b, b_lim))
        names = cbf.names + limit_names

        return ConstraintSet(A=A, b=b, names=names, distances=cbf.distances)
