from pathlib import Path

import numpy as np
import uaibot as ub
from scipy.linalg import expm


_EPS = 1e-12


def propagate_htm(htm: np.ndarray, twist: np.ndarray, dt: float) -> np.ndarray:
    """Propagate an SE(3) pose using the spatial twist [v; w]."""
    htm = np.asarray(htm, dtype=float).reshape(4, 4)
    twist = np.asarray(twist, dtype=float).reshape(6, 1)

    position = htm[:3, 3].reshape(3, 1)
    rotation = htm[:3, :3]
    linear_velocity = twist[:3]
    angular_velocity = twist[3:6]

    next_htm = np.eye(4)
    next_htm[:3, :3] = expm(_skew(angular_velocity) * dt) @ rotation
    next_htm[:3, 3] = (position + linear_velocity * dt).ravel()
    return next_htm


def load_htm_path(file_path: str | Path) -> list[np.ndarray]:
    file_path = Path(file_path)
    htms = []
    current = []

    with file_path.open("r") as file:
        for line in file:
            line = line.strip()
            if not line:
                if current:
                    htms.append(np.asarray(current, dtype=float))
                    current = []
                continue

            current.append([float(value) for value in line.split()])

    if current:
        htms.append(np.asarray(current, dtype=float))

    return htms


def draw_path(path, simulation, color: str = "white", radius: float = 0.02) -> None:
    points = [np.asarray(htm, dtype=float)[:3, 3] for htm in path]
    simulation.add(ub.PointCloud(size=radius, color=color, points=points))


def save_simulation(simulation, address: str | Path, file_name: str) -> None:
    try:
        import utils as uaibot_legacy_utils
    except ImportError:
        uaibot_legacy_utils = None

    # UAIBot uses both import paths internally when generating simulation code.
    original_url_check = ub.Utils.is_url_available
    original_legacy_url_check = (
        uaibot_legacy_utils.Utils.is_url_available
        if uaibot_legacy_utils is not None
        else None
    )

    try:
        ub.Utils.is_url_available = staticmethod(lambda _url, _types: "ok!")
        if uaibot_legacy_utils is not None:
            uaibot_legacy_utils.Utils.is_url_available = staticmethod(
                lambda _url, _types: "ok!"
            )
        simulation.save(address=str(address), file_name=file_name)
    finally:
        ub.Utils.is_url_available = staticmethod(original_url_check)
        if uaibot_legacy_utils is not None:
            uaibot_legacy_utils.Utils.is_url_available = staticmethod(
                original_legacy_url_check
            )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )


def _vee_so3(matrix: np.ndarray) -> np.ndarray:
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]], dtype=float)


def _log_so3(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = np.arccos(cosine)

    if angle < 1e-8:
        return _vee_so3(0.5 * (rotation - rotation.T))

    if np.pi - angle < 1e-6:
        A = 0.5 * (rotation + np.eye(3))
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))

        if rotation[2, 1] - rotation[1, 2] < 0:
            axis[0] = -axis[0]
        if rotation[0, 2] - rotation[2, 0] < 0:
            axis[1] = -axis[1]
        if rotation[1, 0] - rotation[0, 1] < 0:
            axis[2] = -axis[2]

        norm_axis = np.linalg.norm(axis)
        if norm_axis < _EPS:
            axis = _vee_so3((rotation - rotation.T) / (2.0 * np.sin(angle)))
            norm_axis = np.linalg.norm(axis)
            if norm_axis < _EPS:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis /= norm_axis
        else:
            axis /= norm_axis

        return axis * angle

    axis = _vee_so3((rotation - rotation.T) / (2.0 * np.sin(angle)))
    return axis * angle


def _inv_left_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    angle = np.linalg.norm(phi)
    W = _skew(phi)

    if angle < 1e-8:
        return np.eye(3) - 0.5 * W + (1.0 / 12.0) * (W @ W)

    half_angle = 0.5 * angle
    cot_half = np.cos(half_angle) / np.sin(half_angle)
    angle_sq = angle * angle

    return (
        np.eye(3)
        - 0.5 * W
        + (1.0 / angle_sq) * (1.0 - 0.5 * angle * cot_half) * (W @ W)
    )


def log_se3(htm: np.ndarray) -> np.ndarray:
    """SE(3) logarithm kept compatible with the original error metric."""
    htm = np.asarray(htm, dtype=float).reshape(4, 4)
    rotation = htm[:3, :3]
    position = htm[:3, 3]

    phi = _log_so3(rotation)
    v = _inv_left_jacobian_so3(phi) @ position

    algebra = np.zeros((4, 4))
    algebra[:3, :3] = _skew(phi)
    algebra[:3, 3] = v
    return algebra


def inverse_htm(htm: np.ndarray) -> np.ndarray:
    """Inverse of a homogeneous transformation using ndarray semantics."""
    htm = np.asarray(htm, dtype=float).reshape(4, 4)
    rotation = htm[:3, :3]
    position = htm[:3, 3]

    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ position)
    return inverse


def pose_error_norm(H: np.ndarray, H_target: np.ndarray) -> float:
    relative = inverse_htm(H) @ np.asarray(H_target, dtype=float).reshape(4, 4)
    return float(np.linalg.norm(log_se3(relative)))
