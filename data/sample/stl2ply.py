#!/usr/bin/env /usr/bin/python3
 
import open3d as o3d
import numpy as np
import copy
from scipy.spatial.transform import Rotation as rot

names=["outer_mast_AX-VT-Fusion", "outer_mast_AX-VT_inverse-Fusion", "middle_mast_outside_AX-VT-Fusion"]

def gen_ply(name: str):
    mesh = o3d.io.read_triangle_mesh(name+".stl")
    mesh.compute_vertex_normals()
    pcd=mesh.sample_points_uniformly(number_of_points=1000000)
    pcdd=pcd.voxel_down_sample(voxel_size=1.0)
    o3d.visualization.draw_geometries([pcdd])

    diameter = np.linalg.norm(
        np.asarray(pcdd.get_max_bound()) - np.asarray(pcdd.get_min_bound()))
    camera = np.array([-100, -100, 100])
    radius = diameter * 100

    _, pt_map = pcdd.hidden_point_removal(camera, radius)

    pcd = pcdd.select_by_index(pt_map)

    o3d.io.write_point_cloud(name+".ply",pcd)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    for name in names:
        gen_ply(name)