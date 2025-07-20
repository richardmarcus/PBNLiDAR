import numpy as np
path = "/home/oq55olys/Projects/neural_rendering/LiDAR4D/combined_log_paper/kitti360_lidar4d_f1538_0000_baseline/results/"

file= "test_default_ep0500_0001_intensity.npy"
img = np.load(path + file)


#convert to uint8

img*=4

#img greater than 1 is sqrt(img
img[img>1] = np.sqrt(img[img>1])

#normalize to 0-255
img = img / np.max(img) * 255
img = img.astype(np.uint8)

#save to png
from PIL import Image
im = Image.fromarray(img)
im.save(path + file.replace(".npy", ".png"))