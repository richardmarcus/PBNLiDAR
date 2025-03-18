import os
import cv2
import numpy as np

from utils.convert import pano_to_lidar_with_intensities, lidar_to_pano_with_intensities, pano_to_lidar_with_intensities, lidar_to_pano_with_intensities


source_pcd_path = "/media/oq55olys/chonk/Datasets/kittilike/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/"
out_panorama_path = "/media/oq55olys/chonk/Datasets/kittilike/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/panorama/"

avg_img = cv2.imread("full_avg_pano.png", cv2.IMREAD_GRAYSCALE)

#if path exists skip
if os.path.exists(avg_img):

    if not os.path.exists(out_panorama_path):
        os.makedirs(out_panorama_path)

    pcds = os.listdir(source_pcd_path)
    pcds.sort()

    avg_pano = np.zeros((64, 1024), dtype=np.float32)

    for pcd in pcds:
        pcd_path = os.path.join(source_pcd_path, pcd)
        points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)

        depth, pano = lidar_to_pano_with_intensities(points, 64, 1024, [2.02984126984, 11.0317460317, -8.799812, 16.541], max_depth=80, z_offsets = [-0.202, -0.121])

        pano_path = os.path.join(out_panorama_path, pcd.replace(".bin", ".png"))
        avg_pano += pano
        cv2.imwrite(pano_path, pano*255)
        print("Saved panorama to", pano_path)


    avg_pano /= len(pcds)
else:
    avg_pano = cv2.imread("full_avg_pano.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
#average each line but ignore black pixels for weighting
for i in range(64):
    avg_pano[i] = np.average(avg_pano[i][avg_pano[i] > 0])

#avg_all
avg_all = np.average(avg_pano[avg_pano > 0])
avg_pano /= avg_all

#save as np float
avg_pano = avg_pano.astype(np.float32)
np.save("avg_pano.npy", avg_pano)
#get first from out_panorama_path
pano_first = cv2.imread(out_panorama_path + os.listdir(out_panorama_path)[0], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
cv2.imwrite("avg_pano_first.png", pano_first*255)
#multiply with avg_pano
pano_first /= (avg_pano)
#save
cv2.imwrite("avg_pano_first_normalized.png", pano_first*255)


cv2.imwrite("avg_pano.png", avg_pano*200)



