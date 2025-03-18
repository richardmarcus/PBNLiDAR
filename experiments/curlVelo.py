import os

from kitti360scripts.viewer import kitti360Viewer3DRaw


#import open3d
import numpy as np
import open3d as o3d

base_path = "/home/oq55olys/Projects/neural_rendering/LiDAR4D/data/kitti360/KITTI-360/"
path= "/home/oq55olys/Projects/neural_rendering/LiDAR4D/data/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/"

out_path = "/home/oq55olys/Projects/neural_rendering/LiDAR4D/data/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data_curled/"

m_path = "/home/oq55olys/Projects/neural_rendering/LiDAR4D/data/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data_deskewed/"

if not os.path.exists(out_path):
    os.makedirs(out_path)

seq = 0

velo = kitti360Viewer3DRaw.Kitti360Viewer3DRaw(mode='velodyne', seq=seq, path=base_path)

files = os.listdir(path)
files.sort()

for frame in range(1538, len(files)):
    points = velo.loadVelodyneData(frame)
    # curl velodyne
    points2 = velo.curlVelodyneData(frame, points)

    points3 = np.fromfile(os.path.join(m_path, files[frame]), dtype=np.float32).reshape(-1,3)

    points = points[:,0:3]
    points2 = points2[:,0:3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)

    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(points3)

    #paint blue and red
    pcd.paint_uniform_color([0, 0, 1])
    pcd2.paint_uniform_color([1, 0, 0])
    pcd3.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([pcd, pcd2, pcd3])
    exit()





    #write bin file
    #out_file = out_path + files[frame]
    #points.tofile(out_file)

    