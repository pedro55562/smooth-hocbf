import numpy as np


_EPS = 1e-12


def skew(w):
    w = np.asarray(w, dtype=float).reshape(3)
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return np.array(
        [[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]],
        dtype=float,
    )


def vee_so3(W):
    return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=float)


def log_SO3(R):
    R = np.asarray(R, dtype=float).reshape(3, 3)
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    th = np.arccos(c)

    if th < 1e-8:
        return vee_so3(0.5 * (R - R.T))

    if np.pi - th < 1e-6:
        A = (R + np.eye(3)) * 0.5
        axis = np.empty(3, dtype=float)
        axis[0] = np.sqrt(max(A[0, 0], 0.0))
        axis[1] = np.sqrt(max(A[1, 1], 0.0))
        axis[2] = np.sqrt(max(A[2, 2], 0.0))

        if R[2, 1] - R[1, 2] < 0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0:
            axis[2] = -axis[2]

        norm_axis = np.linalg.norm(axis)
        if norm_axis < _EPS:
            axis = vee_so3((R - R.T) / (2.0 * np.sin(th)))
            norm_axis = np.linalg.norm(axis)
            axis = (
                np.array([1.0, 0.0, 0.0])
                if norm_axis < _EPS
                else axis / norm_axis
            )
        else:
            axis = axis / norm_axis
        return axis * th

    axis = vee_so3((R - R.T) / (2.0 * np.sin(th)))
    return axis * th


def inv_jac_left_SO3(phi):
    phi = np.asarray(phi, dtype=float).reshape(3)
    th = np.linalg.norm(phi)
    W = skew(phi)

    if th < 1e-8:
        return np.eye(3) - 0.5 * W + (1.0 / 12.0) * (W @ W)

    half = 0.5 * th
    cot_half = np.cos(half) / np.sin(half)
    return (
        np.eye(3)
        - 0.5 * W
        + (1.0 / (th * th))
        * (1.0 - th * 0.5 * cot_half)
        * (W @ W)
    )


def log_SE3(H):
    H = np.asarray(H, dtype=float).reshape(4, 4)
    R = H[:3, :3]
    p = H[:3, 3]

    phi = log_SO3(R)
    v = inv_jac_left_SO3(phi) @ p

    A = np.zeros((4, 4), dtype=float)
    A[:3, :3] = skew(phi)
    A[:3, 3] = v
    return np.matrix(A)
