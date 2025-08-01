import cv2
import numpy as np
import os
import time

#input paramater for scene_id
#load from args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--scene_id", type=str, default="0000", help="scene id")
args = parser.parse_args()
scene_id = args.scene_id+"/"

#mask image with h w
mask_image = np.zeros((64, 1024), dtype=np.float32)
base_mask_path = "./masks/base_mask.png"
#convert mask to bool
base_mask = cv2.imread(base_mask_path, cv2.IMREAD_GRAYSCALE)
base_mask = base_mask > 0
base_mask = base_mask.astype(np.float32)
path = "./data/kitti360/train_"+scene_id
all_files = os.listdir(path)
#only take .npy
all_files = [file for file in all_files if file.endswith(".npy")]
#sort
all_files = sorted(all_files)
#only take 
num_files = len(all_files)
k=1
for file in all_files:
    #read npy it is shape 1024, 64, 3 and take 3rd channel
    pano = np.load(path + file)
    mask = (pano[:, :, 0]> 0) & (pano[:, :, 1] > 0) & (pano[:, :, 2] > 0)
    mask_image += (~mask).astype(np.float32)
    cv2.imwrite("intensity_mask.png", mask_image/num_files*255)
    k+=1
    #sleep
    time.sleep(0.001)


mask_image = mask_image / num_files
cv2.imwrite(path + "intensity_mask.png", mask_image*255)
#use base mask to filter the mask image

mask_image = mask_image * (1-base_mask)
cv2.imwrite(path + "filtered_intensity_mask.png", mask_image*255)

mask_image = mask_image > 0.2
mask_image = mask_image.astype(np.uint8)
cv2.imwrite(path + "advanced_intensity_mask.png", mask_image*255)
