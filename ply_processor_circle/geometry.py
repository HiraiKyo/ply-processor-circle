import math

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def normalize(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    """_summary_

    Args:
        vector (NDArray[np.float32]): _description_

    Returns:
        NDArray[np.float32]: _description_
    """
    if np.linalg.norm(vector) == 0:
        return vector

    return vector / np.linalg.norm(vector)


def rotation_xyz(pointcloud, theta_x, theta_y, theta_z):
    theta_x = math.radians(theta_x)
    theta_y = math.radians(theta_y)
    theta_z = math.radians(theta_z)
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta_x), -math.sin(theta_x)],
            [0, math.sin(theta_x), math.cos(theta_x)],
        ]
    )

    rot_y = np.array(
        [
            [math.cos(theta_y), 0, math.sin(theta_y)],
            [0, 1, 0],
            [-math.sin(theta_y), 0, math.cos(theta_y)],
        ]
    )

    rot_z = np.array(
        [
            [math.cos(theta_z), -math.sin(theta_z), 0],
            [math.sin(theta_z), math.cos(theta_z), 0],
            [0, 0, 1],
        ]
    )

    rot_matrix = rot_z.dot(rot_y.dot(rot_x))
    rot_pointcloud = rot_matrix.dot(pointcloud.T).T
    return rot_pointcloud, rot_matrix


def point_line_distance(
    points: NDArray[np.float32],
    line_point: NDArray[np.float32],
    line_vector: NDArray[np.float32],
) -> NDArray[np.float32]:
    """_summary_

    Args:
        point (NDArray[np.float32]): _description_
        line (np.ndarray(1, 6)): _description_

    Returns:
        float: _description_
    """
    u = points - line_point
    v = normalize(line_vector)
    vt = np.inner(u, v).reshape(-1, 1).dot(v.reshape(-1, 3))
    return np.linalg.norm(u - vt, axis=1)


def point_plane_distance(
    point: NDArray[np.float32],
    plane_model: NDArray[np.float32],
) -> float:
    """_summary_

    Args:
        points (NDArray[np.float32]): _description_
        plane_model (NDArray[np.float32]): _description_

    Returns:
        float: _description_
    """
    a, b, c, d = plane_model
    return np.abs(a * point[0] + b * point[1] + c * point[2] + d) / np.sqrt(
        a**2 + b**2 + c**2
    )


def get_rotation_matrix_from_vectors(vec1, vec2):
    """_summary_

    Args:
        vec1: _description_
        vec2: _description_

    Returns:
        _description_
    """
    # vec1 -> vec2 の回転ベクトルを導出
    b = normalize(vec1)
    a = normalize(vec2)
    cross = np.cross(a, b)
    dot = np.dot(a, b)
    angle = np.arccos(dot)
    rotvec = normalize(cross) * angle

    # 回転行列を導出
    rotation_matrix = Rotation.from_rotvec(rotvec).as_matrix()

    return rotation_matrix
