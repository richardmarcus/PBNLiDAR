import torch
import numpy as np
import cv2
import os
from data.base_dataset import get_lidar_rays
from utils.convert import  lidar_to_pano_with_intensities, calculate_ring_ids, pano_to_lidar_with_intensities, compare_lidar_to_pano_with_intensities
from mpl_toolkits.mplot3d import Axes3D
from data.preprocess.kitti360_loader import KITTI360Loader
from matplotlib import pyplot as plt
from utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from concurrent.futures import ThreadPoolExecutor

from model.runner import interpolate_poses


device = "cuda" if torch.cuda.is_available() else "cpu"

path= "/home/oq55olys/Projects/neural_rendering/LiDAR4D/data/kitti360/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/"

path_mc = "/media/oq55olys/chonk/Datasets/kittilike/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data_deskewed/"

cam_path = "/home/oq55olys/Projects/neural_rendering/LiDAR4D/data/kitti360/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/"

kitti_360_root = "/home/oq55olys/Projects/neural_rendering/LiDAR4D/data/kitti360/KITTI-360"
# Load KITTI-360 dataset.
k3 = KITTI360Loader(kitti_360_root)
sequence_name = "2013_05_28_drive_0000"

skip = 1
plt.figure(figsize=(10, 10))
num_seq = 1000
fid = 1545#1780

num_merge = num_seq*2

interpolate_jump = 1

print("core frame ", fid)

first_frame = fid - num_seq
frame_ids = list(range(first_frame, first_frame+num_merge))
# Get lidar2world.
lidar2world = k3.load_lidars(sequence_name, frame_ids)
pose_before = k3.load_lidars(sequence_name, [first_frame-1-interpolate_jump])[0]
pose_after = k3.load_lidars(sequence_name, [first_frame+num_merge+interpolate_jump])[0]
cam_to_world_dict, p_rect_00, cam_00_to_velo, r_rect_00 = k3.load_cameras(sequence_name, frame_ids)

mc_poses_path = "/media/oq55olys/chonk/Datasets/kittilike/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/icp_poses.txt"
#format is 1.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00 0.000000000000000000e+00
mc_poses = []

print("reading mc poses")
with open(mc_poses_path, "r") as f:
    for line in f:
        line = line.strip().split(" ")
        line = [float(i) for i in line]
        pose = np.array(line).reshape(3, 4)
        pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
        mc_poses.append(pose)


files = os.listdir(path)
files.sort()

use_icp = False

big_pcd= []
big_intensities = []
base_pose = lidar2world[0]
base_mc_pose = mc_poses[first_frame]
for file in files:
    if int(file.split(".")[0]) < first_frame:
        continue
    bin_pcd = np.fromfile(os.path.join(path, file), dtype=np.float32).reshape(-1,4)
    intensity = bin_pcd[:,3]
    bin_pcd = bin_pcd[:, :3]
    if use_icp or True:
        mc_pcd = np.fromfile(os.path.join(path_mc, file), dtype=np.float32).reshape(-1,3)
        bin_pcd = mc_pcd


    #downsample 
    bin_pcd = bin_pcd[::skip]
    intensity = intensity[::skip]
    #remove points in a 1m radius
    #init empty mask
    mask = np.ones(bin_pcd.shape[0], dtype=bool)
    mask = np.linalg.norm(bin_pcd, axis=1) > 2.6
    #also mask distance greater than 80m and lower than 2m
    mask = mask & (np.linalg.norm(bin_pcd, axis=1) < 80) &  (bin_pcd[:, 2] > -2)
    bin_pcd = bin_pcd[mask]
    intensity = intensity[mask]
    big_intensities.append(intensity)


    if False:
                #add homogenous coordinates
        bin_pcd = np.concatenate([bin_pcd, np.ones((bin_pcd.shape[0], 1))], axis=1)

        pose = lidar2world[len(big_pcd)]
        if use_icp:
            pose = mc_poses[len(big_pcd)+first_frame]
            base_pose = base_mc_pose
        #bring into base pose coordinates
        pose = np.linalg.inv(base_pose) @ pose
        bin_pcd = pose @ bin_pcd.T
        #remove homogenous coordinates
        bin_pcd = bin_pcd.T[:, :3]
    big_pcd.append(bin_pcd)
    print("add frameid" , file)

    if len(big_pcd) == num_merge:
        break

big_pcd = np.concatenate(big_pcd)
big_intensities = np.concatenate(big_intensities)
big_pcd = np.concatenate([big_pcd, big_intensities[:, None]], axis=1)


big_distances = np.linalg.norm(big_pcd[:, :3], axis=1)

#calculate average distance for every 1m
distances = np.arange(1, 80, 1)
average_intensities = []
for d in distances:

    mask = (big_distances > d) & (big_distances < d+1)
    if np.sum(mask) == 0:
        average_intensities.append(0)
    else:
        #filter out intensities bigger than 0.9
        mask = mask & (big_intensities < 0.6)
        mean_intensity = np.mean(big_pcd[mask, 3])
        normalization = 1#(d-12)/40
        average_intensities.append(mean_intensity*normalization)


plt.plot(distances, average_intensities)
plt.xlabel("Distance [m]")
plt.ylabel("Intensity")
#make ylabel go from 0 to 1
plt.ylim([0, 1])
plt.savefig("intensity_vs_distance.png")


exit()

#vis with open3d    
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(big_pcd[:, :3])
big_col = np.zeros_like(big_pcd[:, :3]) 
big_col[:, 0] = big_intensities
big_col[:, 1] = big_intensities
big_col[:, 2] = big_intensities

pcd.colors = o3d.utility.Vector3dVector(big_col)
o3d.visualization.draw_geometries([pcd])
