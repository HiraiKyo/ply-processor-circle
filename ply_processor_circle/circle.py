import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from sklearn import linear_model
from sklearn.cluster import DBSCAN

from ply_processor_circle.config import Config
from ply_processor_circle.plane import detect_plane
from ply_processor_circle.utils.log import Logger

logger = Logger()
logger.init(Config())


def detect(points_raw: NDArray[np.float64], config=None) -> tuple:
    if config is not None:
        Config.load_config(Config(), config=config)

    # 平面検出
    _, _, plane_model = detect_plane(points_raw)

    # 平面上の1点を原点、Z軸を平面の法線ベクトルとする座標系に変換
    transformation_matrix = np.eye(4)
    mean = np.mean(points_raw, axis=0)
    origin = np.array(
        [
            mean[0],
            mean[1],
            -(plane_model[3] + mean[0] * plane_model[0] + mean[1] * plane_model[1])
            / plane_model[2],
        ]
    )
    points = points_raw - origin
    
    transformation_matrix[:3, :3] = get_rotation_matrix_from_vectors(
        np.array([0, 0, 1]), plane_model[:3]
    )
    transformation_matrix_inv = np.linalg.inv(transformation_matrix)
    # アフィン変換のために4x1に変換
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = np.dot(transformation_matrix, points.T).T
    # XY軸に射影
    points = np.dot(points, np.array([x_axis, y_axis, z_axis]).T)

    clustering = DBSCAN(eps=0.1, min_samples=5).fit(points)
    labels = clustering.labels_

    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_cluster = unique_labels[np.argmax(counts[1:]) + 1]

    circle_points = points[labels == largest_cluster]

    ransac = linear_model.RANSACRegressor()
    X = circle_points[:, :2]
    y = np.sum(X**2, axis=1)
    ransac.fit(X, y)

    center_x, center_y = ransac.estimator_.coef_ / 2

    return center_x, center_y


if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("data/sample/sample.ply")
    detect(pcd)
