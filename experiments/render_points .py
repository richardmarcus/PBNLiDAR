#load .npy point cloud with open3d and render it
import open3d as o3d
import numpy as np
import os

path = "/home/oq55olys/Projects/neural_rendering/LiDAR4D/log/kitti360_lidar4d_f4950_release/simulation/points/"
path2= "/home/oq55olys/Projects/neural_rendering/LiDAR4D/data/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/"

files = os.listdir(path)   
files.sort()
files2 = os.listdir(path2)
files2.sort()
#zip files
files = zip(files, files2)

for fileab in files:
    pcds = []
    for file in fileab:
        if file.endswith(".npy"):
            #read npy with numpy
            points= np.load(path + file)
            color = (0,1,0)
        elif file.endswith(".bin"):
            #get filenumber as int
            filenumber = int(file.split(".")[0])
            filenumber+=4950
            file = str(filenumber).zfill(10) + ".bin"
            #read bin with numpy
            points = np.fromfile(path2 + file, dtype=np.float32).reshape(-1,4)
            color = (1,0,0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        intensity = points[:,3]
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)

