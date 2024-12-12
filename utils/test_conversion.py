import numpy as np
from convert import pano_to_lidar_with_intensities, lidar_to_pano_with_intensities


source_pcd_path = "/media/oq55olys/chonk/Datasets/kittilike/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/"
npy_path = "/home/oq55olys/Projects/neural_rendering/LiDAR4D/log/combine_dataset/"
npy_path ="/home/oq55olys/Projects/neural_rendering/4dbase/LiDAR4D/log/kitti360_lidar4d_f1538_release/simulation/points/"
npy_path ="/home/oq55olys/Projects/neural_rendering/LiDAR4D/log/kitti360_lidar4d_1538_debug/simulation/points/"

pcd_file = source_pcd_path + "0000001538.bin"

fov_lidar = [2, 13.45, -11.45, 13.45]
z_offsets = [-0.5, 0]

lidar_H = 64
lidar_W = 1024


#compare pcd with result after lidar_to_pano and back with pano_to_lidar
# fov_up, fov, fov_up2, fov2 = lidar_K

points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)
#lidar_to_pano_with_intensities(
#    local_points_with_intensities: np.ndarray,
#    lidar_H: int,
#    lidar_W: int,
#    lidar_K: int,
#    max_depth=80,
#    z_offsets = [-0.202, -0.121]
#):

pano, intensities = lidar_to_pano_with_intensities(points, lidar_H, lidar_W, fov_lidar, max_depth=80, z_offsets = z_offsets)

#pano_to_lidar_with_intensities(pano: np.ndarray, intensities, lidar_K, z_offsets):

points_reconstructed = pano_to_lidar_with_intensities(pano, intensities, fov_lidar, z_offsets)

#vis with open3d
import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
#paint orange
pcd.colors = o3d.utility.Vector3dVector(points[:, 3:4] * np.array([1, 0.5, 0]))
#paint uniform orange
pcd.paint_uniform_color([1, 0.5, 0])

pcd_re = o3d.geometry.PointCloud()
pcd_re.points = o3d.utility.Vector3dVector(points_reconstructed[:, :3])
#paint blue
pcd_re.colors = o3d.utility.Vector3dVector(points_reconstructed[:, 3:4] * np.array([0, 0.5, 1]))
#paint uniform blue
pcd_re.paint_uniform_color([0, 0.5, 1])
pcd_npy = o3d.geometry.PointCloud()
#npy_points = np.load(npy_path + "0000001583.npy")
npy_points = np.load(npy_path + "lidar4d_0000.npy")
pcd_npy.points = o3d.utility.Vector3dVector(npy_points[:, :3])
#paint green
pcd_npy.colors = o3d.utility.Vector3dVector(npy_points[:, 3:4] * np.array([0, 1, 0]))
#paint uniform green
pcd_npy.paint_uniform_color([0, 1, 0])

o3d.visualization.draw_geometries([pcd, pcd_re])



