import os
import cv2
import numpy as np


base_path = "/home/oq55olys/Projects/neural_rendering/LiDAR4D/log/kitti360_lidar4d_f1728_release"
#take base path as argument

folders = ["simulation00/", "simulation25/", "simulation50/", "simulation75/"]
#stitching refers to creating a video from 4 seperate image sets that have different interpolation offsets
# Preallocate a list for image sets
image_sets = [[] for _ in range(4)]

# Load all images in one loop for efficiency
for i, folder in enumerate(folders):
    path = os.path.join(base_path, folder)
    #add images/ to the path
    path = os.path.join(path, "images")
    files = sorted([f for f in os.listdir(path) if f.endswith(".png")])  # Filter and sort in one step
    image_sets[i] = [cv2.imread(os.path.join(path, file)) for file in files]  # Load images using a 


# Zip the image sets and flatten them in one go to stitch the images
stitched_images = [img for imgs in zip(*image_sets) for img in imgs]
# Convert images to RGB if they are not already
stitched_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in stitched_images]
#convert to numpy array
stitched_images_rgb = np.array(stitched_images_rgb)

import imageio

# Specify the output video file and frames per second
output_video_path = "stitched_video.mp4"
fps = 10  # 25 ms per frame = 40 frames per second


imageio.mimwrite(
    output_video_path,
    stitched_images_rgb,
    fps=fps, # change frame rate here
    quality=8,
    macro_block_size=1,
)

print(f"Video saved at {output_video_path}")