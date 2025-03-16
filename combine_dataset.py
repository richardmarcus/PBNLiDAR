import os
import shutil
import open3d as o3d
import cv2
import numpy as np

from utils.convert import pano_to_lidar_with_intensities, lidar_to_pano_with_intensities, pano_to_lidar_with_intensities, lidar_to_pano_with_intensities


base_path = "/home/oq55olys/Projects/neural_rendering/LiDAR4D/log/"
sub_paths = ["kitti360_lidar4d_f1538_release", "kitti360_lidar4d_f1728_release", "kitti360_lidar4d_f1908_release", "kitti360_lidar4d_f3353_release"]

out_path = "/home/oq55olys/Projects/neural_rendering/LiDAR4D/log/combine_dataset4k/"

out_path = "/home/oq55olys/Projects/neural_rendering/LiDAR4D/log/kitti360_lidar4d_f1538_release/simulation/points/"
out_img_path = "/home/oq55olys/Projects/neural_rendering/LiDAR4D/log/kitti360_lidar4d_f1538_release/simulation/images/"
vanilla_out_path = "/home/oq55olys/Projects/neural_rendering/4dbase/LiDAR4D/log/kitti360_lidar4d_f1538_release/simulation/points/"
rgb_path = "/media/oq55olys/chonk/Datasets/kittilike/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/"

source_pcd_path = "/media/oq55olys/chonk/Datasets/kittilike/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/"

dataset_dir = "/media/oq55olys/chonk/Datasets/kittilike/KITTI-360/img2img/"
dataset_rgb_dir = dataset_dir + "rgb/"
dataset_intensity_dir = dataset_dir + "intensity4/"

npy_img_path = "/home/oq55olys/Projects/neural_rendering/LiDAR4D/data/kitti360/train/"






if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
if not os.path.exists(dataset_rgb_dir):
    os.makedirs(dataset_rgb_dir)
if not os.path.exists(dataset_intensity_dir):
    os.makedirs(dataset_intensity_dir)


def to_img(points):

    #remove points with z < 0
    points = points[points[:,0] > 0]

    # P_rect_00: Projektionsmatrix
    P_rect_00 = np.array([[552.554261, 0.000000, 682.049453, 0.000000],
                        [0.000000, 552.554261, 238.769549, 0.000000],
                        [0.000000, 0.000000, 1.000000, 0.000000],
                        [0.000000, 0.000000, 0.000000, 1.000000]])


    # Transformation von Kamera zu Velodyne
    # calib_path = "calibration/calib_cam_to_velo.txt"
    calib_cam_to_velodyne = np.array([[0.04307104361, -0.08829286498, 0.995162929, 0.8043914418],
                        [-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],
                        [-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824],
                        [0.000000, 0.000000, 0.000000, 1.000000]])
    # Invertieren
    calib_velodyne_to_cam = np.linalg.inv(calib_cam_to_velodyne)
    # in homogene Koordinaten umwandeln (X, Y, Z, 1)
    points_homogeneous = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))

    # Transformation in Kamerakoordinaten
    transformed_points = np.dot(points_homogeneous, calib_velodyne_to_cam.T)

    # 3D Punkte in 2D projizieren
    image_points = np.dot(transformed_points, P_rect_00.T)
    # Dehomogenisierung
    image_points[:, 0] /= image_points[:, 2]
    image_points[:, 1] /= image_points[:, 2]



    image = np.zeros((376, 1408), dtype=np.uint8)

    intensity = points[:, 3]
    #add 0.1 and clip to 1
    intensity = np.clip(intensity + 0.1, 0, 1)

    mask = (image_points[:, 0] >= 0) & (image_points[:, 0] < 1408) & (image_points[:, 1] >= 0) & (image_points[:, 1] < 376)
    image_points = image_points[mask]
    points = points[mask]
    intensity = intensity[mask]
    u = image_points[:, 0].astype(np.uint16)
    v = image_points[:, 1].astype(np.uint16)
    image[v, u] =(intensity* 255).astype(np.uint8)
    return image

if False and not os.path.exists(out_path):
  os.makedirs(out_path)

  for sub_path in sub_paths:
      path = base_path + sub_path
      #add /points/
      points_path = path + "/simulation/points/"

      #extract the first file id
      file_id = int(sub_path.split("_")[-2][1:])
      print("file_id: ", file_id)

      #copy point files to new folder but construct the filename as file_id+ i + .npy
      #0000001919.npy for example so also make sure the number of digits is the same
      files = os.listdir(points_path)
      #sort
      files.sort()
      for i, file in enumerate(files):
          shutil.copy(points_path + file, out_path + str(file_id + i).zfill(10) + ".npy")
          print("copying: ", file_id + i)

    


#for pcd in outpath ply : project to image and save as intensity
pcd_files = os.listdir(out_path)
#sort
pcd_files.sort()

pcd_bin_files = os.listdir(source_pcd_path)
#sort
pcd_bin_files.sort()

#only take #pcd_files from pcd_bin_files
pcd_bin_files = pcd_bin_files[1538:1538+len(pcd_files)]


for pcd_file, pcd_file_bin in zip(pcd_files, pcd_bin_files):
    print("processing: ", pcd_file, pcd_file_bin)


    #pcd_file = pcd_file_bin
  
    #if ending isply

 
    fov_lidar = [2.02984126984, 11.0317460317, -8.799812, 16.541]
    z_offsets = [-0.202, -0.121]
    #fov_lidar = [2, 13.45, -11.45, 13.45]
    #z_offsets = [-0.0, 0]
    points_bin = np.fromfile(source_pcd_path + pcd_file_bin, dtype=np.float32).reshape(-1, 4)
    depth_bin, pano_bin = lidar_to_pano_with_intensities(points_bin, 64, 1024, fov_lidar, max_depth=80, z_offsets =z_offsets)
    #
    # z_offsets = [0.0, 0]
    points = np.load(out_path + pcd_file)
    depth, pano = lidar_to_pano_with_intensities(points, 64, 1024, fov_lidar, max_depth=80, z_offsets =z_offsets)

    points_vanilla = np.load(vanilla_out_path + pcd_file)
    depth_vanilla, pano_vanilla = lidar_to_pano_with_intensities(points_vanilla, 64, 1024, fov_lidar, max_depth=80, z_offsets =z_offsets)

    img = to_img(points)
    img_bin = to_img(points_bin)
    img_vanilla = to_img(points_vanilla)
    npy_img_file = pcd_file_bin.split(".")[0] + ".npy"
    #load image it has size of 1024 , (64*3) ,3  = 1024 , 192 , 3 and uint8
    img_gt = np.load(npy_img_path + npy_img_file)

    #get second channel
    img_gt = img_gt[:,:,1]


    if False:
        pcd_bin = o3d.geometry.PointCloud()
        pcd_bin.points = o3d.utility.Vector3dVector(points_bin[:, :3])
        #use green intensity
        pcd_bin.colors = o3d.utility.Vector3dVector(np.stack((np.zeros(points_bin.shape[0]), points_bin[:, 3], np.zeros(points_bin.shape[0])), axis=-1))


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        #use red intensity
        pcd.colors = o3d.utility.Vector3dVector(np.stack((points[:, 3], np.zeros(points.shape[0]), np.zeros(points.shape[0])), axis=-1))
      
        o3d.visualization.draw_geometries([pcd, pcd_bin])

  
    rgb_img = cv2.imread(rgb_path + pcd_file_bin.split(".")[0] + ".png")
    #convert rgb to gray
    rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    #convert gray to rgb by stacking
    rgb_img = np.stack((rgb_gray, rgb_gray, rgb_gray), axis=-1)
    #darken rgb img
    rgb_img = rgb_img * 0.2

    #save as png
    img_path = dataset_intensity_dir + pcd_file_bin.split(".")[0] + ".png"

    pano_direct = cv2.imread(out_img_path + pcd_file.split(".")[0] + ".png")
    intensity_direct = pano_direct[64:128]
    intensity_direct = cv2.cvtColor(intensity_direct, cv2.COLOR_BGR2GRAY)

    depth_direct = pano_direct[128:192]    

    
    pcd_direct = pano_to_lidar_with_intensities(depth_direct, intensity_direct, fov_lidar, z_offsets)
    pano_re, intensity_re = lidar_to_pano_with_intensities(pcd_direct, 64, 1024, fov_lidar, max_depth=80, z_offsets =z_offsets)


    #pano_direct are 3 images stacked above each other and rgb
    #only keep the middle one

    #convert to gray

    #create rgb with pano shape
    pano_rgb= np.zeros((pano.shape[0], pano.shape[1], 3), dtype=np.uint8)
    #use pano as r and pano_bin as g
    pano_rgb[:,:,2] = pano*255
    pano_rgb[:,:,1] = intensity_direct
    pano_rgb[:,:,0] = intensity_re*255
    #pano_rgb[:,:,1] = pano_bin*255
    #pano_rgb[:,:,0] = pano_vanilla*255
    #pano_rgb[:,:,0] = img_gt*255

    #only keep middle 20 percent
    #pano_rgb = pano_rgb[:, 312:712]
 
    #resisze to 1408 x 256
    #pano_rgb = cv2.resize(pano_rgb, (1408, 256))

    depth_rgb = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    depth_rgb[:,:,2] = depth*3
    #depth_rgb[:,:,1] = depth_bin*3
    depth_rgb[:,:,0] = depth_vanilla*3

    #only keep middle 20 percent
    depth_rgb = depth_rgb[:, 312:712]
    #resisze to 1408 x 256
    depth_rgb = cv2.resize(depth_rgb, (1408, 256))


 
    def overwrite_cross_pixels(img, rgb_img, channel, value):
        for i in range(rgb_img.shape[0]):
            for j in range(rgb_img.shape[1]):
                if img[i, j] > 0:
                    # Modify the center pixel
                    rgb_img[i, j, channel] = value[i,j]
                    
                    # Modify the pixels above, below, left, and right (cross pattern)
                    if i - 1 >= 0:
                        rgb_img[i - 1, j, channel] = value[i,j]  # Above
                    if i + 1 < rgb_img.shape[0]:
                        rgb_img[i + 1, j, channel] = value[i,j]  # Below
                    if j - 1 >= 0:
                        rgb_img[i, j - 1, channel] = value[i,j]  # Left
                    if j + 1 < rgb_img.shape[1]:
                        rgb_img[i, j + 1, channel] = value[i,j]  # Right

    # Overwrite green channel (index 1) based on img using cross pattern
    #overwrite_cross_pixels(img, rgb_img, 2, img)

    # Overwrite red channel (index 0) based on img_bin using cross pattern
    #overwrite_cross_pixels(img_bin, rgb_img, 1, img_bin)

    #overwrite_cross_pixels(img_vanilla, rgb_img, 0, img_vanilla)

                   
    #concatenate pano_rgb to rgb_img
    #rgb_img = np.concatenate((rgb_img, pano_rgb), axis=0)
    #rgb_img = np.concatenate((rgb_img, depth_rgb), axis=0)

    rgb_img = pano_rgb

    cv2.imwrite(img_path, rgb_img)
    print("saved: ", img_path)

    exit()


exit()

